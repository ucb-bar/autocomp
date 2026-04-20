import pathlib
import re
import subprocess
import time
from typing import List

from autocomp.common import logger, HARNESSES_DIR, SOLS_DIR
from autocomp.search.prob import Prob
from autocomp.backend.eval_backend import EvalBackend


RUNNER_DIR = pathlib.Path(__file__).parent / "runner"
TMP_DIR = pathlib.Path(__file__).parent / "tmp_files"
COMPILE_TIMEOUT = 60
RUN_TIMEOUT = 120
COOLDOWN_SECONDS = 2  # pause between candidates to mitigate thermal throttling


def extract_metal_code(response: str) -> str:
    """Extract .metal kernel source from an agent response.

    Handles:
    - Markdown fenced code blocks (```metal, ```c, ```cpp, or bare ```)
    - Raw Metal source (returned as-is if no fences found)
    """
    # Try to extract from markdown fenced blocks
    pattern = r'```(?:metal|c|cpp|c\+\+)?\s*\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        # Return the longest match (likely the full kernel, not a snippet)
        return max(matches, key=len).strip()

    # If no fences, check if it looks like Metal source already
    stripped = response.strip()
    if '#include' in stripped or 'kernel void' in stripped or 'using namespace metal' in stripped:
        return stripped

    return stripped


class MetalEvalBackend(EvalBackend):
    def __init__(self, compile_timeout=COMPILE_TIMEOUT, run_timeout=RUN_TIMEOUT):
        super().__init__()
        self.compile_timeout = compile_timeout
        self.run_timeout = run_timeout
        self.build_dir = RUNNER_DIR / "build"
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        self.build_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str = "") -> List[dict]:
        results = []
        for i, code_str in enumerate(code_strs):
            if i > 0:
                time.sleep(COOLDOWN_SECONDS)
            results.append(self._evaluate_single(prob, code_str, i))
        return results

    def _get_problem_name(self, prob: Prob) -> str:
        """Derive the problem name from prob_type and prob_id.

        Looks for the harness .cpp file matching the problem ID.
        """
        harness_dir = HARNESSES_DIR / prob.prob_type
        matches = list(harness_dir.glob(f"{prob.prob_id}_*.cpp"))
        if matches:
            return matches[0].stem
        return f"{prob.prob_id}"

    def _build_harness(self, prob: Prob) -> tuple[bool, str]:
        """Build the harness binary for this problem. Returns (success, error_msg)."""
        problem_name = self._get_problem_name(prob)
        harness_dir = HARNESSES_DIR / prob.prob_type
        binary = self.build_dir / problem_name

        # Check if binary exists and is newer than source
        harness_src = harness_dir / f"{problem_name}.cpp"
        if binary.exists() and binary.stat().st_mtime > harness_src.stat().st_mtime:
            return True, ""

        try:
            result = subprocess.run(
                ["make", "-C", str(RUNNER_DIR), "harness",
                 f"PROBLEM={problem_name}",
                 f"HARNESS_DIR={harness_dir}"],
                capture_output=True, text=True, timeout=self.compile_timeout,
            )
            if result.returncode != 0:
                return False, f"Harness build failed:\n{result.stderr}"
            return True, ""
        except subprocess.TimeoutExpired:
            return False, "Harness build timed out"

    def _compile_kernel(self, metal_path: pathlib.Path) -> tuple[bool, str, pathlib.Path]:
        """Compile a .metal file to .metallib. Returns (success, error_msg, metallib_path)."""
        metallib_name = metal_path.stem + ".metallib"
        metallib_path = self.build_dir / metallib_name

        try:
            result = subprocess.run(
                ["make", "-C", str(RUNNER_DIR), "metallib",
                 f"KERNEL_SRC={metal_path}"],
                capture_output=True, text=True, timeout=self.compile_timeout,
            )
            if result.returncode != 0:
                return False, f"Metal compile failed:\n{result.stderr}", metallib_path
            return True, "", metallib_path
        except subprocess.TimeoutExpired:
            return False, "Metal compilation timed out", metallib_path

    def _get_reference_metallib(self, prob: Prob) -> tuple[bool, str, pathlib.Path]:
        """Compile the reference solution kernel. Returns (success, error_msg, metallib_path)."""
        sol_dir = SOLS_DIR / prob.prob_type
        sol_files = list(sol_dir.glob(f"{prob.prob_id}_*.metal"))
        if not sol_files:
            return False, f"No reference .metal file found for prob {prob.prob_id}", pathlib.Path()

        ref_metal = sol_files[0]
        # _compile_kernel produces <stem>.metallib in build_dir
        ref_metallib = self.build_dir / f"{ref_metal.stem}.metallib"

        # Cache: skip if already built and newer than source
        if ref_metallib.exists() and ref_metallib.stat().st_mtime > ref_metal.stat().st_mtime:
            return True, "", ref_metallib

        return self._compile_kernel(ref_metal)

    def _run_harness(self, prob: Prob, candidate_metallib: pathlib.Path,
                     reference_metallib: pathlib.Path) -> tuple[str, str]:
        """Run the harness binary. Returns (stdout, stderr)."""
        problem_name = self._get_problem_name(prob)
        binary = self.build_dir / problem_name

        try:
            result = subprocess.run(
                [str(binary), str(candidate_metallib), str(reference_metallib)],
                capture_output=True, text=True, timeout=self.run_timeout,
            )
            return result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return "", "Harness execution timed out"

    def _parse_output(self, stdout: str) -> dict:
        """Parse structured output from the harness binary."""
        result = {"correct": False, "latency": None, "stddev": None}

        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("CORRECT:"):
                result["correct"] = line.split(":", 1)[1].strip() == "true"
            elif line.startswith("MEDIAN_MS:"):
                try:
                    result["latency"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("STDDEV_MS:"):
                try:
                    result["stddev"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("ERROR:"):
                result["error"] = line.split(":", 1)[1].strip()

        return result

    def _evaluate_single(self, prob: Prob, code_str: str, idx: int) -> dict:
        """Evaluate a single Metal kernel candidate."""
        # Extract clean Metal code from agent response
        metal_code = extract_metal_code(code_str)

        # Write to temp file
        metal_path = TMP_DIR / f"candidate_{idx}.metal"
        metal_path.write_text(metal_code)

        # Build harness (cached)
        ok, err = self._build_harness(prob)
        if not ok:
            logger.error(f"Harness build failed: {err}")
            return {"correct": False, "latency": None, "stdout": "", "stderr": err}

        # Compile candidate kernel
        ok, err, candidate_metallib = self._compile_kernel(metal_path)
        if not ok:
            logger.error(f"Kernel compile failed: {err}")
            return {"correct": False, "latency": None, "stdout": "", "stderr": err}

        # Compile reference kernel (cached)
        ok, err, reference_metallib = self._get_reference_metallib(prob)
        if not ok:
            logger.error(f"Reference kernel compile failed: {err}")
            return {"correct": False, "latency": None, "stdout": "", "stderr": err}

        # Run harness: candidate vs reference
        stdout, stderr = self._run_harness(prob, candidate_metallib, reference_metallib)
        if stderr and not stdout:
            logger.error(f"Harness run failed: {stderr}")
            return {"correct": False, "latency": None, "stdout": stdout, "stderr": stderr}

        # Parse output
        result = self._parse_output(stdout)
        result["stdout"] = stdout
        result["stderr"] = stderr
        return result
