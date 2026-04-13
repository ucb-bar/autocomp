"""
JAXBench eval backend for autocomp.

Evaluates generated implementations against JAXBench workloads on TPU.
Implementations must define a `workload(*inputs)` function matching the
signature of the original workload in the JAXBench benchmark file.

Reuses TpuHardwareBackend for SSH/SCP transport to the TPU VM.
"""
import ast
import json
import os
import pathlib
import re
import secrets
import subprocess
from datetime import datetime
from typing import List

from autocomp.common import logger
from autocomp.search.prob import Prob
from autocomp.backend.tpu.tpu_eval import TpuHardwareBackend, _ensure_tpu_vm_running

_THIS_DIR = pathlib.Path(__file__).resolve().parent
RUNNER_SCRIPT = _THIS_DIR / "jaxbench_runner.py"

_jaxbench_env = os.getenv("JAXBENCH_DIR", "")
JAXBENCH_DIR = pathlib.Path(_jaxbench_env) if _jaxbench_env else _THIS_DIR.parent.parent.parent.parent / "JAXBench"
BENCHMARK_DIR = JAXBENCH_DIR / "benchmark"

# Maps prob_type to which file to use as the workload (baseline or optimized)
_VARIANT_FOR_PROB_TYPE = {
    "jaxbench-pallas": "optimized",
    "jaxbench-baseline": "baseline",
    "jaxbench": "baseline",
}

DELIM_START = "===JAXBENCH_IMPL_START==="
DELIM_END = "===JAXBENCH_IMPL_END==="


def _find_workload_file(prob: Prob) -> pathlib.Path:
    """Locate the JAXBench workload .py file for a Prob.

    prob_id should be the workload directory name (e.g., "7p_Ragged_Paged_Attention").
    prob_type selects the variant: "jaxbench-pallas" -> optimized.py, others -> baseline.py.
    """
    variant = _VARIANT_FOR_PROB_TYPE.get(prob.prob_type, "baseline")
    workload_dir = BENCHMARK_DIR / str(prob.prob_id)
    if not workload_dir.is_dir():
        # Try fuzzy match: prob_id might be a suffix like "ragged_paged_attention"
        for d in BENCHMARK_DIR.iterdir():
            if d.is_dir() and str(prob.prob_id).lower().replace("_", "") in d.name.lower().replace("_", ""):
                workload_dir = d
                break
    target = workload_dir / f"{variant}.py"
    if not target.exists():
        target = workload_dir / "baseline.py"
    if not target.exists():
        raise FileNotFoundError(
            f"No workload file found for {prob.prob_type}/{prob.prob_id} "
            f"(tried {workload_dir / f'{variant}.py'} and {workload_dir / 'baseline.py'})"
        )
    return target


def extract_workload_code(prob: Prob) -> str:
    """Return a minimal workload snippet for the LLM (no harness boilerplate).

    For Model-style files: extracts the forward body into a
    standalone ``workload()`` function with input shapes as comments.
    For workload-style files: returns imports, CONFIG,
    create_inputs, and the workload function only.
    """
    import textwrap

    path = _find_workload_file(prob)
    source = path.read_text()
    tree = ast.parse(source)

    has_model = any(
        isinstance(n, ast.ClassDef) and n.name == "Model" for n in tree.body
    )

    if has_model:
        return _extract_model_style(source, tree)
    return _extract_workload_style(source, tree)


def _extract_model_style(source: str, tree: ast.Module) -> str:
    """Convert Model.forward + get_inputs into a standalone workload snippet."""
    import textwrap
    lines = source.splitlines()

    # Collect top-level assignments (M, N, batch_size, etc.)
    assignments = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            assignments.append(ast.get_source_segment(source, node))

    # Extract Model.__init__ args (for get_init_inputs) and forward body
    forward_src = None
    forward_args = []
    init_params = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    init_params = [
                        a.arg for a in item.args.args if a.arg != "self"
                    ]
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    forward_args = [
                        a.arg for a in item.args.args if a.arg != "self"
                    ]
                    body_start = item.body[0].lineno - 1
                    body_end = item.end_lineno
                    raw = "\n".join(lines[body_start:body_end])
                    forward_src = textwrap.dedent(raw)

    if forward_src is None:
        return source

    # Extract get_inputs body to build shape comments
    input_comment = ""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_inputs":
            seg = ast.get_source_segment(source, node)
            if seg:
                input_comment = f"# Input setup (for reference):\n# {seg.replace(chr(10), chr(10) + '# ')}\n"

    parts = ["import jax", "import jax.numpy as jnp", ""]
    if assignments:
        parts.extend(a for a in assignments if a)
        parts.append("")
    if input_comment:
        parts.append(input_comment)
    args_str = ", ".join(forward_args)
    parts.append(f"def workload({args_str}):")
    for line in forward_src.splitlines():
        parts.append(f"    {line}" if line.strip() else "")
    return "\n".join(parts) + "\n"


def _extract_workload_style(source: str, tree: ast.Module) -> str:
    """Keep everything from a workload file except benchmark() and __main__ blocks."""
    lines = source.splitlines()
    skip_ranges = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "benchmark":
            skip_ranges.append((node.lineno - 1, node.end_lineno))
        elif isinstance(node, ast.If):
            test = node.test
            if (isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"):
                skip_ranges.append((node.lineno - 1, node.end_lineno))

    kept = []
    skip_set = set()
    for start, end in skip_ranges:
        for i in range(start, end):
            skip_set.add(i)

    for i, line in enumerate(lines):
        if i not in skip_set:
            kept.append(line)

    return "\n".join(kept) + "\n"


def _extract_latency(text: str) -> float | None:
    for line in text.split("\n"):
        if "Latency:" in line and "ms" in line:
            try:
                return float(line.split("Latency:")[-1].split("ms")[0].strip())
            except ValueError:
                continue
    return None


def _extract_util(text: str) -> float | None:
    for line in text.split("\n"):
        m = re.search(r"(?i)\b(utilization|util)\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(%)?", line)
        if m:
            val = float(m.group(2))
            if m.group(3) is None and 0.0 <= val <= 1.0:
                val *= 100.0
            if 0.0 <= val <= 1000.0:
                return round(val, 3)
    return None


class JaxBenchEvalBackend(TpuHardwareBackend):
    """Evaluate implementations against JAXBench workloads on TPU."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._jax_setup_done = False
        self._runner_uploaded = False

    # ── public API ───────────────────────────────────────────────────────────

    def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> List[dict]:
        self.ensure_tpu_vm()
        workload_path = _find_workload_file(prob)
        logger.info(
            "Evaluating %d implementation(s) on TPU for JAXBench %s/%s (%s)",
            len(code_strs), prob.prob_type, prob.prob_id, workload_path.name,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = _THIS_DIR / "eval_outputs" / timestamp
        eval_dir.mkdir(parents=True, exist_ok=True)

        results = self._run_evaluation(workload_path, code_strs, eval_dir)
        if results is None:
            logger.info("Batch failed, falling back to individual evaluation")
            results = []
            for code in code_strs:
                single = self._run_evaluation(workload_path, [code], eval_dir)
                results.append(
                    single[0] if single
                    else {"correct": False, "latency": None, "stdout": "", "stderr": "eval failed"}
                )

        for idx, r in enumerate(results):
            if r["correct"]:
                logger.info("Implementation %d/%d: %.3f ms", idx + 1, len(code_strs), r["latency"])
            else:
                logger.info("Implementation %d/%d: FAIL", idx + 1, len(code_strs))

        return results

    def ensure_tpu_vm(self) -> None:
        if self._transport_mode() != "gcloud":
            return
        if self._tpu_vm_checked:
            return
        _ensure_tpu_vm_running(
            tpu_name=self.tpu_name,
            zone=self.tpu_zone,
            accelerator_type=os.getenv("AUTOCOMP_TPU_ACCELERATOR_TYPE") or "v6e-1",
            version=os.getenv("AUTOCOMP_TPU_RUNTIME_VERSION") or "v2-alpha-tpuv6e",
            project=self.tpu_project,
        )
        self._tpu_vm_checked = True

    # ── internal ─────────────────────────────────────────────────────────────

    def _run_evaluation(
        self,
        workload_path: pathlib.Path,
        code_strs: list[str],
        eval_dir: pathlib.Path,
    ) -> list[dict] | None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + secrets.token_hex(4)
        remote_dir = f"/tmp/autocomp_jaxbench/{run_id}"

        # 0. Create remote directory
        self._ssh(f"mkdir -p {remote_dir}")

        # 1. Upload runner script (once per session)
        if not self._runner_uploaded:
            if self._scp(RUNNER_SCRIPT, "jaxbench_runner.py") != 0:
                return None
            self._runner_uploaded = True

        # 2. Upload workload file
        remote_workload = f"{remote_dir}/workload.py"
        if self._scp(workload_path, remote_workload) != 0:
            return None

        # 3. Write and upload each implementation
        impl_remote_paths = []
        for i, code in enumerate(code_strs):
            local = eval_dir / f"impl_{run_id}_{i}.py"
            local.write_text(code, encoding="utf-8")
            remote = f"{remote_dir}/impl_{i}.py"
            if self._scp(local, remote) != 0:
                return None
            impl_remote_paths.append(remote)

        # 4. Build remote command
        impl_args = " ".join(impl_remote_paths)
        setup_cmd = self._jax_setup_command() if not self._jax_setup_done else ""
        run_python = f"{self._python_bin} jaxbench_runner.py {remote_workload} {impl_args}"

        stdout_f = f"{remote_dir}/stdout.txt"
        stderr_f = f"{remote_dir}/stderr.txt"

        remote_cmd = (
            f"{setup_cmd}"
            f"{run_python} > {stdout_f} 2> {stderr_f}; true"
        )

        # 5. Execute — scale timeout: runner has a 120s per-impl timeout,
        # plus overhead for JAX setup, reference compilation, and SSH.
        ssh_timeout = 180 + 150 * len(code_strs)
        try:
            self._ssh(remote_cmd, timeout=ssh_timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Remote execution timed out after %ds", ssh_timeout)
            return None

        stdout = self._ssh_cat(stdout_f)
        stderr = self._ssh_cat(stderr_f)

        (eval_dir / f"output_{run_id}.txt").write_text(
            f"=== STDOUT ===\n{stdout}\n=== STDERR ===\n{stderr}",
            encoding="utf-8",
        )

        if DELIM_START not in stdout:
            logger.warning("No implementation output produced. stderr:\n%s", stderr[:500])
            return None

        self._jax_setup_done = True
        return self._parse_output(stdout, stderr, len(code_strs))

    def _parse_output(self, stdout: str, stderr: str, num_impls: int) -> list[dict]:
        sections = stdout.split(DELIM_START)[1:]  # skip preamble
        impl_blocks = []
        for section in sections:
            end = section.find(DELIM_END)
            impl_blocks.append(section[:end].strip() if end != -1 else section.strip())

        results = []
        for idx in range(num_impls):
            result = {"correct": False, "latency": None, "stdout": "", "stderr": stderr, "util": None}
            if idx < len(impl_blocks):
                block = impl_blocks[idx]
                result["stdout"] = block

                # Try structured JSON first (emitted by runner)
                for line in reversed(block.split("\n")):
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            parsed = json.loads(line)
                            result["correct"] = parsed.get("correct", False)
                            result["latency"] = parsed.get("latency")
                            break
                        except json.JSONDecodeError:
                            pass

                # Fall back to regex extraction
                if result["latency"] is None:
                    result["latency"] = _extract_latency(block)
                    if result["latency"] is not None and "FAIL" not in block and "ERROR" not in block:
                        result["correct"] = True

                result["util"] = _extract_util(block)
            results.append(result)
        return results

    # ── SSH/SCP helpers (thin wrappers around TpuHardwareBackend) ─────────

    def _jax_setup_command(self) -> str:
        check = f"{self._python_bin} -c 'import jax; assert jax.__version__==\"0.9.2\", jax.__version__' >/dev/null 2>&1"
        install = f"{self._python_bin} -m pip install -U 'jax[tpu]==0.9.2' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q"
        return f"({check}) || ({install}) 2>&1; "

    def _scp(self, local_path: pathlib.Path, remote_path: str) -> int:
        dest = (
            f"{self.tpu_name}:{remote_path}"
            if self._transport_mode() == "gcloud"
            else f"{self._ssh_target()}:{remote_path}"
        )
        cmd = self._build_scp_cmd(source=str(local_path), dest=dest)
        logger.debug("scp: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
        if proc.returncode != 0:
            logger.error("scp failed (exit %s): %s", proc.returncode, proc.stderr[:200])
        return proc.returncode

    def _ssh(self, remote_command: str, timeout: int = 600) -> subprocess.CompletedProcess:
        cmd = self._build_ssh_cmd(remote_command=remote_command, allocate_tty=False, batch_mode=True)
        logger.debug("ssh: %s", " ".join(cmd))
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)

    def _ssh_cat(self, remote_path: str) -> str:
        cmd = self._build_ssh_cmd(
            remote_command=f"cat {remote_path} 2>/dev/null || true",
            allocate_tty=False, batch_mode=True,
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        return proc.stdout or ""
