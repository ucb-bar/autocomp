import subprocess
import pathlib
import multiprocessing
import shutil
import time
import glob
import re
from typing import List

from autocomp.common import logger, SOLS_DIR
from autocomp.search.prob import Prob
from autocomp.backend.hardware_backend import HardwareBackend

# Environment path variables
SATURN_CHIPYARD_PATH = "/scratch/charleshong/rvv/chipyard"
SATURN_ZEPHYR_BASE = "/scratch/charleshong/rvv/zephyr-chipyard-sw"  # Zephyr installation root
SATURN_ZEPHYR_APP_PATH = "/scratch/charleshong/rvv/zephyr-chipyard-sw/samples/rvv_bench"  # Contains src/main.c, CMakeLists.txt, prj.conf
SATURN_TEMP_DIR =  "/scratch/charleshong/rvv/saturn_tmp"

# Timeouts (seconds)
SATURN_SPIKE_TIMEOUT = 60.0
SATURN_COMPILE_TIMEOUT = 120
SATURN_FIRESIM_TIMEOUT = 300.0
SATURN_FIRESIM_INDIVIDUAL_TIMEOUT = 500.0


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


def build_single(code_contents: str, candidate_idx: int, timestamp: str, return_dict: dict):
    """
    Build a single candidate in an isolated directory using Zephyr.

    Results are stored in return_dict with keys:
        - "binary": pathlib.Path on success, None on failure
        - "error": error string on failure, None on success
        - "work_dir": path to the work directory
    """
    unique_id = f"{timestamp}_candidate{candidate_idx}"
    work_dir = pathlib.Path(SATURN_TEMP_DIR) / f"saturn_build_{unique_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    return_dict["work_dir"] = str(work_dir)

    try:
        # Copy Zephyr app template to isolated directory
        app_template = pathlib.Path(SATURN_ZEPHYR_APP_PATH)
        app_dir = work_dir / "app"
        shutil.copytree(
            app_template,
            app_dir,
            ignore=shutil.ignore_patterns('build*', '__pycache__', '*.o', '*.elf')
        )

        # Write code to isolated copy
        test_file = app_dir / "src" / "main.c"
        test_file.write_text(code_contents)

        # Compile with Zephyr - each build has isolated dirs so parallel is safe
        build_dir = work_dir / "build"
        build_cmd = f"""
            cd {SATURN_ZEPHYR_BASE} && \
            source scripts/set_envvars_sdk.sh && \
            source tools/miniforge3/etc/profile.d/conda.sh && \
            conda activate zephyr && \
            west build -p -b spike_riscv64 -d {build_dir} {app_dir}
        """
        result = subprocess.run(
            ["bash", "-c", build_cmd],
            capture_output=True,
            timeout=SATURN_COMPILE_TIMEOUT
        )
        if result.returncode != 0:
            return_dict["binary"] = None
            return_dict["error"] = f"Compile error: {result.stderr.decode()}"
            return

        binary = build_dir / "zephyr" / "zephyr.elf"
        if not binary.exists():
            return_dict["binary"] = None
            return_dict["error"] = f"Compile error: binary not found at {binary}"
            return

        return_dict["binary"] = str(binary)
        return_dict["error"] = None

    except subprocess.TimeoutExpired:
        return_dict["binary"] = None
        return_dict["error"] = "Compile timeout"
    except Exception as e:
        return_dict["binary"] = None
        return_dict["error"] = f"Build error: {str(e)}"


def run_spike_on_binary(binary_path: pathlib.Path, return_dict: dict):
    
    try:
        result = subprocess.run(
            ["spike", f"--isa=rv64gcv_zicntr", str(binary_path)],
            capture_output=True,
            text=True,
            errors="ignore"
        )
        return_dict["retval"] = result.stdout
    except Exception as e:
        return_dict["retval"] = f"Spike error: {str(e)}"


def run_spike_mp(code_contents_lst: list[str], timeout: float = SATURN_SPIKE_TIMEOUT) -> list[str]:
    """
    Build using Zephyr and run spike, both in parallel.

    Phase 1: Build all candidates in parallel (isolated directories)
    Phase 2: Run spike on all binaries in parallel (read-only)
    """
    results = ["Error"] * len(code_contents_lst)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Phase 1: Parallel builds
    logger.info("Building %d candidates in parallel...", len(code_contents_lst))
    manager = multiprocessing.Manager()
    build_dicts = []
    build_procs = []

    for code_i, code_contents in enumerate(code_contents_lst):
        return_dict = manager.dict()
        build_dicts.append((code_i, return_dict))
        p = multiprocessing.Process(
            target=build_single,
            args=(code_contents, code_i, timestamp, return_dict)
        )
        p.start()
        build_procs.append(p)

    # Wait for builds with timeout
    build_timeout = SATURN_COMPILE_TIMEOUT
    start = time.time()
    while time.time() - start <= build_timeout:
        if not any(p.is_alive() for p in build_procs):
            break
        time.sleep(0.1)
    else:
        logger.warning("Build phase exceeded timeout, terminating remaining builds.")
        for p in build_procs:
            if p.is_alive():
                p.terminate()
                p.join()

    # Collect build results
    binary_paths = []
    for i, (code_i, return_dict) in enumerate(build_dicts):
        if return_dict.get("binary"):
            binary_paths.append((code_i, pathlib.Path(return_dict["binary"])))
        else:
            results[code_i] = return_dict.get("error", "Build failed")

    logger.info("Built %d/%d candidates successfully", len(binary_paths), len(code_contents_lst))

    if not binary_paths:
        return results

    # Phase 2: Parallel spike execution
    logger.info("Running spike on %d binaries in parallel...", len(binary_paths))
    manager = multiprocessing.Manager()
    return_dicts = []
    procs = []

    for code_i, binary_path in binary_paths:
        return_dict = manager.dict()
        return_dicts.append((code_i, return_dict))
        p = multiprocessing.Process(
            target=run_spike_on_binary,
            args=(binary_path, return_dict)
        )
        p.start()
        procs.append(p)

    # Polling loop with timeout
    start = time.time()
    while time.time() - start <= timeout:
        if not any(p.is_alive() for p in procs):
            break
        time.sleep(0.1)
    else:
        logger.info("Spike ran for more than %d seconds, terminating.", timeout)
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join()

    # Collect results
    for i, (code_i, return_dict) in enumerate(return_dicts):
        if procs[i].exitcode != 0 and "retval" not in return_dict:
            results[code_i] = "Timeout"
        else:
            results[code_i] = return_dict.get("retval", "Error")

    return results


def build_firesim_binary(code_contents: str, timestamp: str) -> pathlib.Path | str:
    """
    Build a single binary for FireSim execution.

    Returns binary path on success, error string on failure.
    """
    unique_id = f"{timestamp}_firesim"
    work_dir = pathlib.Path(SATURN_TEMP_DIR) / f"saturn_firesim_{unique_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Copy Zephyr app template
        app_template = pathlib.Path(SATURN_ZEPHYR_APP_PATH)
        app_dir = work_dir / "app"
        shutil.copytree(
            app_template,
            app_dir,
            ignore=shutil.ignore_patterns('build*', '__pycache__', '*.o', '*.elf')
        )

        # Write code
        test_file = app_dir / "src" / "main.c"
        test_file.write_text(code_contents)

        # Compile with Zephyr for FireSim target
        build_dir = work_dir / "build"
        build_cmd = f"""
            cd {SATURN_ZEPHYR_BASE} && \
            source scripts/set_envvars_sdk.sh && \
            source tools/miniforge3/etc/profile.d/conda.sh && \
            conda activate zephyr && \
            west build -p -b spike_riscv64 -d {build_dir} {app_dir}
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
    p = subprocess.Popen(
        ["bash", "-c", infrasetup_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    with p.stdout:
        for line in p.stdout:
            logger.debug(line.decode().strip())
    p.wait()

    if p.returncode != 0:
        logger.error("firesim infrasetup failed with return code %d", p.returncode)
        return {}

    # 3. Run workload with timeout
    logger.info("Running `firesim runworkload`")
    runworkload_cmd = f"{firesim_setup} && firesim runworkload"
    p = subprocess.Popen(
        ["bash", "-c", runworkload_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    try:
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.terminate()
        p.wait()
        logger.warning("FireSim runworkload timed out after %d seconds.", timeout)
        # Kill any remaining FireSim processes
        logger.info("Running `firesim kill`")
        kill_cmd = f"{firesim_setup} && firesim kill"
        subprocess.run(
            ["bash", "-c", kill_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        return {}

    if p.returncode != 0:
        logger.error("firesim runworkload failed with return code %d", p.returncode)
        return {}

    logger.info("FireSim runworkload finished")
    time.sleep(2)  

    current_results_dirs = sorted(results_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True) if results_dir.exists() else []
    new_dirs = [d for d in current_results_dirs if d not in orig_results_dirs]

    if not new_dirs:
        logger.error("No new results directory found after running FireSim.")
        return {}
#
    relevant_logs = []
    for dir in new_dirs:
        dir_logs = glob.glob(f"{dir}/*/uartlog", recursive=True)
        relevant_logs.extend(dir_logs)

    if not relevant_logs:
        logger.error("No uartlog found after running FireSim.")
        return {}

    if len(relevant_logs) > 1:
        logger.warning("Multiple logs found, using first one: %s", relevant_logs)

    logger.info("Parsing FireSim results from: %s", relevant_logs[0])
    return parse_firesim_uartlog(relevant_logs[0])


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


def run_firesim_individual(code_contents: str,
                           firesim_path: pathlib.Path,
                           candidate_idx: int,
                           timeout: float = SATURN_FIRESIM_INDIVIDUAL_TIMEOUT) -> int | None:
    """
    Run a single candidate on FireSim (fallback for batch failures).

    Returns latency on success, None on failure.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    binary_result = build_firesim_binary(code_contents, f"{timestamp}_{candidate_idx}")

    if isinstance(binary_result, str):
        logger.warning("Failed to build FireSim binary for candidate %d: %s", candidate_idx, binary_result)
        return None

    results = run_firesim_batch(binary_result, firesim_path, timeout=timeout)
    return results.get(candidate_idx)


class SaturnHardwareBackend(HardwareBackend):
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
        return f"SaturnHardwareBackend(vlen={self.vlen}, elen={self.elen})"

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
            test_output_per_code_str = run_spike_mp(
                test_codes,
                timeout=SATURN_SPIKE_TIMEOUT * len(code_strs)  # Scale timeout with batch size
            )

            # Parse results
            for code_i, test_output in enumerate(test_output_per_code_str):
                if "Correct" in test_output:
                    logger.debug("Code %d, Test %d: Correct result", code_i, test_i)
                    stats[code_i]["test_results"][test_i] = True
                    # stats[code_i]["stdout"] = test_output

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
                logger.warning("FireSim path not configured")
            else:
                passing_indices = [i for i, s in enumerate(stats) if s["correct"]]
                passing_codes = [clean_code_strs[i] for i in passing_indices]

                if not passing_codes:
                    logger.info("No passing candidates to run on FireSim.")
                else:
                    logger.info("Running %d passing candidates on FireSim...", len(passing_codes))

                    try:
                        firesim_latencies = self._run_firesim_batch(
                            passing_codes, passing_indices, prob
                        )

                        # 3. Update stats with FireSim latencies
                        for orig_idx, latency in firesim_latencies.items():
                            if latency is not None:
                                stats[orig_idx]["firesim_latency"] = latency
                                logger.debug("FireSim latency for candidate %d: %d cycles", orig_idx, latency)

                    except Exception as e:
                        logger.warning("FireSim batch failed: %s.", e)
    

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
            passing_codes, passing_indices, first_test
        )

        # Compile for FireSim
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        binary_result = build_firesim_binary(combined_code, timestamp)

        if isinstance(binary_result, str):
            raise RuntimeError(f"Failed to compile FireSim binary: {binary_result}")

        # Run on FireSim
        return run_firesim_batch(
            binary_result,
            self.firesim_path,
            timeout=SATURN_FIRESIM_TIMEOUT
        )

    def _build_firesim_combined_code(self,
                                      passing_codes: list[str],
                                      passing_indices: list[int],
                                      test) -> str:
        """
        Build combined C code with all passing candidates for FireSim batching.

        Inlines each candidate's code with cycle counting around it.
        """
        # Generate inlined code blocks for each candidate
        code_blocks = []
        for code, orig_idx in zip(passing_codes, passing_indices):
            code_blocks.append(f"""
    // Run candidate {orig_idx}
    RESET_STATE(); 
    fence();
    start_cycle = read_cycles();
    {{
{code}
    }}
    fence();
    end_cycle = read_cycles();
    printf("ID {orig_idx} latency: %lu cycles\\n", end_cycle - start_cycle);
""")

        code_blocks_str = "\n".join(code_blocks)

        # Build the substitution block - all code inlined, no function definitions
        substitution = f"""
    int start_cycle, end_cycle;

{code_blocks_str}
"""

        # Use the test's modify_test_code to inject our combined code
        return test.modify_test_code(substitution)


if __name__ == "__main__":
    prob = Prob("qs8", 0)
    files = [SOLS_DIR / "qs8" / "qs8-vaddc.c"]
    if files[0].exists():
        code_str = files[0].read_text()
        code_strs = [code_str, code_str, code_str] 
        backend = SaturnHardwareBackend()
        stats = backend.evaluate_code(prob, code_strs, "firesim")
        for i, stat in enumerate(stats):
            print(f"  Candidate {i}: {stat}")
    else:
        print("No test files found. Create tests in tests/qs8/ and sols in sols/qs8/")