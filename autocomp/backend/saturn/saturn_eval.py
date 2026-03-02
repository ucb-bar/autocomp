import subprocess
import pathlib
import multiprocessing
import shutil
import time
import glob
import re
import os
from typing import List

from autocomp.common import logger, SOLS_DIR
from autocomp.search.prob import Prob
from autocomp.backend.eval_backend import EvalBackend

# Environment path variables
SATURN_CHIPYARD_PATH = "/scratch/charleshong/saturn-tutorial/chipyard"
SATURN_ZEPHYR_BASE = "/scratch/charleshong/saturn-tutorial/zephyr-chipyard-sw"  # Zephyr installation root

# Timeouts (seconds)
SATURN_SPIKE_TIMEOUT = 60.0
SATURN_COMPILE_TIMEOUT = 120.0
SATURN_FIRESIM_TIMEOUT = 500.0

FIRESIM_REPEAT_ITERS = 15

SATURN_TEMP_DIR = pathlib.Path(__file__).parent / "tmp_dir"
SATURN_ZEPHYR_APP_PATH = pathlib.Path(__file__).parent / "rvv_bench" # Contains src/main.c, CMakeLists.txt, prj.conf

def clean_code(code_str: str) -> str:
    """
    Takes LLM-generated code, removes the "test" wrapper function, and cleans up
    anything that is not runnable C code.

    For example:
    '''
    void test(...) {
        // RVV vector code
        vfloat32m1_t va = vle32_v_f32m1(a, vl);
        ...
    }
    '''

    becomes:

    '''
    // RVV vector code
    vfloat32m1_t va = vle32_v_f32m1(a, vl);
    ...
    '''
    """
    if not code_str:
        return ""

    # Find the function wrapper
    if "void test(" not in code_str:
        # No wrapper, return as-is
        return code_str

    after_void_test_str = code_str[code_str.find("void test("):]
    start = after_void_test_str.find('{') + 1
    end = after_void_test_str.rfind('}')
    body = after_void_test_str[start:end]
    return body


MAX_BUILD_SLOTS = min(8,os.cpu_count())


def _build_and_run_spike(args: tuple) -> tuple:
    
    code_contents, slot_id, candidate_idx = args
    slot_dir = SATURN_TEMP_DIR / f"build_slot_{slot_id}"
    app_dir = slot_dir / "app"
    build_dir = slot_dir / "build"


    try:
        is_first = not (build_dir / "zephyr" / "zephyr.elf").exists()

        if is_first:
            if app_dir.exists():
                shutil.rmtree(app_dir)
            if build_dir.exists():
                shutil.rmtree(build_dir)
            shutil.copytree(
                SATURN_ZEPHYR_APP_PATH, app_dir,
                ignore=shutil.ignore_patterns('build*', '__pycache__', '*.o', '*.elf')
            )

        (app_dir / "src" / "main.c").write_text(code_contents)

        pristine_flag = "-p" if is_first else ""
        build_cmd = f"""
            cd {SATURN_ZEPHYR_BASE} && \
            source scripts/set_envvars_sdk.sh && \
            source tools/miniforge3/etc/profile.d/conda.sh && \
            conda activate zephyr && \
            west build {pristine_flag} -b spike_riscv64 -d {build_dir} {app_dir} -DXNNPACK_ENABLE_RISCV_VECTOR=ON -DXNNPACK_ENABLE_RISCV_GEMMINI=OFF 
        """
        result = subprocess.run(
            ["bash", "-c", build_cmd],
            capture_output=True,
            timeout=SATURN_COMPILE_TIMEOUT
        )
        if result.returncode != 0:
            return (candidate_idx, f"Compile error: {result.stderr.decode()}")

        binary = build_dir / "zephyr" / "zephyr.elf"
        if not binary.exists():
            return (candidate_idx, f"Binary not found at {binary}")

    except subprocess.TimeoutExpired:
        return (candidate_idx, "Compile timeout")
    except Exception as e:
        return (candidate_idx, f"Build error: {str(e)}")

    # --- Run spike ---
    try:
        result = subprocess.run(
            ["spike", "--isa=rv64gcv_zicntr", str(binary)],
            capture_output=True,
            text=True,
            errors="ignore",
            timeout=SATURN_SPIKE_TIMEOUT
        )
        return (candidate_idx, result.stdout)
    except subprocess.TimeoutExpired:
        return (candidate_idx, "Timeout")
    except Exception as e:
        return (candidate_idx, f"Spike error: {str(e)}")


def run_spike_mp(code_contents_lst: list[str]) -> list[str]:

    results = ["Error"] * len(code_contents_lst)

    tasks = [(code, i % MAX_BUILD_SLOTS, i) for i, code in enumerate(code_contents_lst)]
    logger.info("Building & running spike on %d candidates across %d slots...",
                len(code_contents_lst), min(len(code_contents_lst), MAX_BUILD_SLOTS))

    with multiprocessing.Pool(MAX_BUILD_SLOTS) as pool:
        for candidate_idx, stdout in pool.imap_unordered(_build_and_run_spike, tasks):
            results[candidate_idx] = stdout

    return results


FIRESIM_BUILD_SLOT = 0  # Reuse spike slot 0 -- spike finishes before firesim starts

def build_firesim_binary(code_contents: str) -> pathlib.Path | str:
   
    slot_dir = SATURN_TEMP_DIR / f"build_slot_{FIRESIM_BUILD_SLOT}"
    app_dir = slot_dir / "app"
    build_dir = slot_dir / "build"

    try:
        is_first = not (build_dir / "zephyr" / "zephyr.elf").exists()

        if is_first:
            if app_dir.exists():
                shutil.rmtree(app_dir)
            if build_dir.exists():
                shutil.rmtree(build_dir)
            shutil.copytree(
                SATURN_ZEPHYR_APP_PATH, app_dir,
                ignore=shutil.ignore_patterns('build*', '__pycache__', '*.o', '*.elf')
            )

        (app_dir / "src" / "main.c").write_text(code_contents)

        pristine_flag = "-p" if is_first else ""
        build_cmd = f"""
            cd {SATURN_ZEPHYR_BASE} && \
            source scripts/set_envvars_sdk.sh && \
            source tools/miniforge3/etc/profile.d/conda.sh && \
            conda activate zephyr && \
            west build {pristine_flag} -b spike_riscv64 -d {build_dir} {app_dir}
        """
        result = subprocess.run(
            ["bash", "-c", build_cmd],
            capture_output=True,
            timeout=SATURN_COMPILE_TIMEOUT
        )
        if result.returncode != 0:
            return f"Compile error: {result.stderr.decode()}"

        binary = build_dir / "zephyr" / "zephyr.elf"
        if not binary.exists():
            return f"Compile error: binary not found at {binary}"

        return binary

    except subprocess.TimeoutExpired:
        return "Compile timeout"
    except Exception as e:
        return f"Build error: {str(e)}"


def run_firesim_batch(binary_path: pathlib.Path,
                      firesim_path: pathlib.Path,
                      timeout: float = SATURN_FIRESIM_TIMEOUT) -> dict[int, int]:
    """
    Run binary on FireSim and parse results.

    """
    results_dir = firesim_path / "deploy" / "results-workload"

    # 1. Copy binary to workload directory
    workload_dir = firesim_path / "deploy" / "workloads" / "saturn"
    workload_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(binary_path, workload_dir / "saturn_test-baremetal")

    # Get current results dirs for comparison
    orig_results_dirs = sorted(results_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True) if results_dir.exists() else []

    # FireSim commands need environment from sourceme-manager.sh
    firesim_setup = f"cd {firesim_path} && source {firesim_path}/sourceme-manager.sh"

    # 2. Run infrasetup
    logger.info("Running `firesim infrasetup`")
    infrasetup_cmd = f"{firesim_setup} && firesim infrasetup"
    infrasetup = subprocess.run(
        ["bash", "-c", infrasetup_cmd],
        capture_output=True,
        text=True
    )

    if infrasetup.returncode != 0:
        logger.error("firesim infrasetup failed with return code %d", infrasetup.returncode)
        logger.error("FireSim stdout:\n%s", infrasetup.stdout or "")
        logger.error("FireSim stderr:\n%s", infrasetup.stderr or "")
        raise RuntimeError(f"firesim infrasetup failed with return code {infrasetup.returncode}")

    # 3. Run workload with timeout
    logger.info("Running `firesim runworkload`")
    runworkload_cmd = f"{firesim_setup} && firesim runworkload"
    try:
        runworkload = subprocess.run(
            ["bash", "-c", runworkload_cmd],
            capture_output=True,
            text=True,
            timeout=timeout
        )
    except subprocess.TimeoutExpired:
        logger.warning("FireSim runworkload timed out after %d seconds.", timeout)
        # Kill any remaining FireSim processes
        logger.info("Running `firesim kill`")
        kill_cmd = f"{firesim_setup} && firesim kill"
        subprocess.run(
            ["bash", "-c", kill_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        raise RuntimeError(f"firesim runworkload timed out after {timeout} seconds")

    if runworkload.returncode != 0:
        logger.error("firesim runworkload failed with return code %d", runworkload.returncode)
        logger.error("FireSim stdout:\n%s", (infrasetup.stdout or "") + (runworkload.stdout or ""))
        logger.error("FireSim stderr:\n%s", (infrasetup.stderr or "") + (runworkload.stderr or ""))
        raise RuntimeError(f"firesim runworkload failed with return code {runworkload.returncode}")

    logger.info("FireSim runworkload finished")
    time.sleep(2)

    current_results_dirs = sorted(results_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True) if results_dir.exists() else []
    new_dirs = [d for d in current_results_dirs if d not in orig_results_dirs]

    if not new_dirs:
        logger.error("No new results directory found after running FireSim.")
        logger.error("FireSim stdout:\n%s", (infrasetup.stdout or "") + (runworkload.stdout or ""))
        logger.error("FireSim stderr:\n%s", (infrasetup.stderr or "") + (runworkload.stderr or ""))
        raise RuntimeError("No new results directory found after running FireSim.")

    relevant_logs = []
    for dir in new_dirs:
        dir_logs = glob.glob(f"{dir}/*/uartlog", recursive=True)
        relevant_logs.extend(dir_logs)

    if not relevant_logs:
        logger.error("No uartlog found after running FireSim.")
        logger.error("FireSim stdout:\n%s", (infrasetup.stdout or "") + (runworkload.stdout or ""))
        logger.error("FireSim stderr:\n%s", (infrasetup.stderr or "") + (runworkload.stderr or ""))
        raise RuntimeError("No uartlog found after running FireSim.")

    if len(relevant_logs) > 1:
        logger.warning("Multiple logs found, using first one: %s", relevant_logs)

    logger.info("Parsing FireSim results from: %s", relevant_logs[0])
    results = parse_firesim_uartlog(relevant_logs[0])
    if not results:
        logger.error("FireSim stdout:\n%s", (infrasetup.stdout or "") + (runworkload.stdout or ""))
        logger.error("FireSim stderr:\n%s", (infrasetup.stderr or "") + (runworkload.stderr or ""))
        raise RuntimeError("No latency results found in FireSim uartlog.")
    return results


def parse_firesim_uartlog(log_path: str) -> dict[int, int]:
    """
    Parse FireSim uartlog to extract latencies per candidate.

    Expected format: "ID <orig_idx> latency: <cycles> cycles"
    """
    results = {}
    log_content = pathlib.Path(log_path).read_text()

    # Pattern: "ID 5 latency: 12345 cycles"
    pattern = r"ID (\d+) latency: (\d+) cycles"
    matches = re.findall(pattern, log_content)

    for orig_idx_str, latency_str in matches:
        orig_idx = int(orig_idx_str)
        latency = int(latency_str)
        results[orig_idx] = latency
        logger.debug("Parsed FireSim result: candidate %d = %d cycles", orig_idx, latency)

    return results


class SaturnEvalBackend(EvalBackend):
    """
    Hardware backend for evaluating RVV code on Saturn core using spike/FireSim.

    """

    def __init__(self,
                 vlen: int = 256,
                 elen: int = 64,
                 chipyard_path: str = None):

        super().__init__()
        self.vlen = vlen
        self.elen = elen

        # Set paths
        chipyard_path = chipyard_path or SATURN_CHIPYARD_PATH
        if chipyard_path:
            self.chipyard_path = pathlib.Path(chipyard_path).resolve()
            self.saturn_path = self.chipyard_path / "generators" / "firechip"
            self.firesim_path = self.chipyard_path / "sims" / "firesim"
        else:
            self.chipyard_path = None
            self.saturn_path = None
            self.firesim_path = None

    def __repr__(self):
        return f"SaturnEvalBackend(vlen={self.vlen}, elen={self.elen})"

    def evaluate_code_spike(self, prob: Prob, code_strs: list[str]) -> List[dict]:
        """Convenience method for spike-only evaluation."""
        return self.evaluate_code(prob, code_strs, "spike")

    def evaluate_code_firesim(self, prob: Prob, code_strs: list[str]) -> List[dict]:
        """Convenience method for FireSim evaluation."""
        return self.evaluate_code(prob, code_strs, "firesim")

    def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> List[dict]:
        """
        Evaluate code candidates and return results.

        """
        if not self.saturn_path or not self.saturn_path.exists():
            raise ValueError(
                "Saturn path not configured. Set SATURN_CHIPYARD_PATH environment variable "
                "or pass chipyard_path to constructor."
            )

        # Initialize results - all passing at the beginning (flip to False on test failure)
        stats = [{
            "correct": True,
            "test_results": {}
        } for _ in code_strs]

        # Clean code strings (remove wrapper functions)
        clean_code_strs = [clean_code(code_str) for code_str in code_strs]
        # Run each test
        for test_i, test in enumerate(prob.tests):
            logger.info("Running spike on %d code candidates for test %d", len(code_strs), test_i)

            # Get test code with each candidate injected
            test_codes = [test.get_test_code([code_str]) for code_str in clean_code_strs]

            # Run spike in parallel
            test_output_per_code_str = run_spike_mp(test_codes)

            # Parse results
            for code_i, test_output in enumerate(test_output_per_code_str):
                if "Correct" in test_output:
                    logger.debug("Code %d, Test %d: Correct result", code_i, test_i)
                    stats[code_i]["test_results"][test_i] = True

                    # Extract latency from spike output
                    if simulator == "spike" and "Generated implementation latency" in test_output:
                        try:
                            latency_str = test_output.split("Generated implementation latency: ")[-1]
                            sol_latency = int(latency_str.split(" cycles")[0])
                            stats[code_i]["latency"] = sol_latency
                        except (ValueError, IndexError):
                            logger.warning("Failed to parse latency from spike output for code %d", code_i)
                else:
                    logger.debug("Code %d, Test %d: Incorrect result", code_i, test_i)
                    stats[code_i]["test_results"][test_i] = False
                    stats[code_i]["correct"] = False
                    stats[code_i]["stderr"] = test_output

                    # Log specific error type
                    if test_output == "Compile error":
                        logger.debug("Code %d: Compilation failed", code_i)
                    elif test_output == "Timeout":
                        logger.debug("Code %d: Spike timeout", code_i)

        # FireSim evaluation - run passing candidates on FireSim for accurate latency
        if simulator == "firesim":
            if not self.firesim_path or not self.firesim_path.exists():
                msg = "FireSim path not configured"
                logger.warning(msg)
                for s in stats:
                    if s["correct"]:
                        s["correct"] = False
                        s["stderr"] = msg
            else:
                passing_indices = [i for i, s in enumerate(stats) if s["correct"]]
                passing_codes = [clean_code_strs[i] for i in passing_indices]

                if not passing_codes:
                    logger.info("No passing candidates to run on FireSim.")
                else:
                    logger.info("Running %d passing candidates on FireSim...", len(passing_codes))

                    firesim_latencies = self._run_firesim_batch(
                        passing_codes, passing_indices, prob
                    )

                    # Update stats with FireSim latencies
                    for orig_idx, latency in firesim_latencies.items():
                        if latency is not None:
                            stats[orig_idx]["firesim_latency"] = latency
                            stats[orig_idx]["latency"] = latency
                            logger.debug("FireSim latency for candidate %d: %d cycles", orig_idx, latency)
                    missing_indices = set(passing_indices) - set(firesim_latencies.keys())
                    if missing_indices:
                        msg = "FireSim did not return latency for candidate"
                        for orig_idx in missing_indices:
                            stats[orig_idx]["correct"] = False
                            stats[orig_idx]["stderr"] = msg

        logger.debug("Evaluation stats: %s", stats)
        return stats

    def _run_firesim_batch(self,
                           passing_codes: list[str],
                           passing_indices: list[int],
                           prob: Prob) -> dict[int, int]:
        """
        Run a batch of passing codes on FireSim.

        """
        # Get test template for concatenation
        if not prob.tests:
            raise ValueError("No tests available for FireSim template")

        first_test = prob.tests[0]

        # Build concatenated test code
        # We use the test's template but with all passing codes as separate functions
        combined_code = self._build_firesim_combined_code(
            passing_codes, passing_indices, first_test, prob
        )

        # Compile for FireSim (uses stable build slot for incremental builds)
        binary_result = build_firesim_binary(combined_code)

        if isinstance(binary_result, str):
            raise RuntimeError(f"Failed to compile FireSim binary: {binary_result}")

        # Run on FireSim
        return run_firesim_batch(
            binary_result,
            self.firesim_path,
            timeout=SATURN_FIRESIM_TIMEOUT
        )

    @staticmethod
    def _extract_candidate_params(code_str: str) -> list[tuple[str, str]]:
        """
        Extract function parameters from the first function definition in a
        solution file. Works with any function name (void test, void xnn_f32_..., etc).

        Returns list of (type, name) tuples. Empty if no function definition found
        (e.g. gemm tests where candidates access globals directly).
        """
        # Find the first function definition: look for `) {` or `)\n{` pattern
        # and work backwards to find the matching `(`
        # Use regex to find: return_type func_name(
        match = re.search(r'\b\w[\w\s*]*\s+(\w+)\s*\(', code_str)
        if not match:
            return []

        # Find the opening paren position
        paren_start = code_str.index('(', match.start())
        after_paren = code_str[paren_start + 1:]

        # Find matching closing paren (handle nested parens)
        paren_depth = 1
        param_end = 0
        for j, ch in enumerate(after_paren):
            if ch == '(':
                paren_depth += 1
            elif ch == ')':
                paren_depth -= 1
                if paren_depth == 0:
                    param_end = j
                    break
        param_str = after_paren[:param_end]

        # Parse comma-separated parameters
        params = []
        for param in param_str.split(','):
            param = ' '.join(param.split())  # normalize whitespace
            if not param:
                continue
            parts = param.rsplit(None, 1)
            if len(parts) == 2:
                var_type, var_name = parts
                if var_name.startswith('*'):
                    var_type += ' *'
                    var_name = var_name[1:]
                params.append((var_type, var_name))

        return params

    def _build_firesim_combined_code(self,
                                      passing_codes: list[str],
                                      passing_indices: list[int],
                                      test,
                                      prob: Prob = None) -> str:
        """
        Build combined C code with all passing candidates for FireSim batching.

        Each candidate is wrapped in its own noinline function to isolate
        vector register allocation (prevents register pressure from LMUL=8
        candidates causing spills/crashes in a single giant function).

        The function signature is derived from the solution file in sols/.
        """
        # Read the solution file to extract function parameters
        params = []
        if prob:
            sol_dir = SOLS_DIR / prob.prob_type
            sol_files = list(sol_dir.glob(f"{prob.prob_id}_*.c"))
            if sol_files:
                sol_content = sol_files[0].read_text()
                params = self._extract_candidate_params(sol_content)

        if params:
            param_sig = ", ".join(f"{t} {n}" for t, n in params)
            param_args = ", ".join(n for _, n in params)
        else:
            param_sig = "void"
            param_args = ""

        # Generate noinline wrapper functions for each candidate
        func_defs = []
        for code, orig_idx in zip(passing_codes, passing_indices):
            func_defs.append(f"""
__attribute__((noinline)) void run_candidate_{orig_idx}({param_sig}) {{
{code}
}}
""")

        func_defs_str = "\n".join(func_defs)

        # Generate call blocks in main
        code_blocks = []
        for code, orig_idx in zip(passing_codes, passing_indices):
            code_blocks.append(f"""
    // Run candidate {orig_idx}
    RESET_STATE();
    fence();
    __asm__ volatile("vsetvli x0, x0, e8, m1, ta, ma");
    start_cycle = read_cycles();
    run_candidate_{orig_idx}({param_args});
    fence();
    __asm__ volatile("vsetvli x0, x0, e8, m1, ta, ma");
    end_cycle = read_cycles();
    printf("ID {orig_idx} latency: %lu cycles\\n", end_cycle - start_cycle);
""")

        code_blocks_str = "\n".join(code_blocks)

        # Build the substitution block
        substitution = f"""
    unsigned long start_cycle, end_cycle;

{code_blocks_str}
"""

        # Use the test's modify_test_code to inject our combined code,
        # then prepend the noinline function definitions before main()
        combined = test.modify_test_code(substitution)

        # Insert function definitions before main()
        main_pos = combined.find("\nint main(")
        if main_pos == -1:
            main_pos = combined.find("\nint main ")
        if main_pos != -1:
            combined = combined[:main_pos] + "\n" + func_defs_str + combined[main_pos:]
        else:
            # Fallback: prepend after includes
            combined = func_defs_str + "\n" + combined

        return combined


if __name__ == "__main__":
    prob = Prob("qs8", 0)
    files = [SOLS_DIR / "qs8" / "qs8-vaddc.c"]
    if files[0].exists():
        code_str = files[0].read_text()
        code_strs = [code_str, code_str, code_str]
        backend = SaturnEvalBackend()
        stats = backend.evaluate_code(prob, code_strs, "firesim")
        for i, stat in enumerate(stats):
            print(f"  Candidate {i}: {stat}")
    else:
        print("No test files found. Create tests in tests/qs8/ and sols in sols/qs8/")
