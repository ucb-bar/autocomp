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
XNNPACK_CHIPYARD_PATH = "/scratch/kchern2/chipyard"
XNNPACK_ZEPHYR_BASE = "/scratch/kchern2/zephyr-chipyard-sw"  # Zephyr installation root

# Timeouts (seconds)
XNNPACK_SPIKE_TIMEOUT = 60.0
XNNPACK_COMPILE_TIMEOUT = 300.0
XNNPACK_FIRESIM_PER_CANDIDATE_TIMEOUT = 10.0
XNNPACK_FIRESIM_MIN_TIMEOUT = 45.0


XNNPACK_TEMP_DIR = pathlib.Path(__file__).parent / "tmp_dir"
XNNPACK_ZEPHYR_APP_PATH = pathlib.Path(__file__).parent / "rvv_bench" # Contains src/main.cpp, CMakeLists.txt, prj.conf
XNNPACK_FIRESIM_SIM_SLOT_DIR = pathlib.Path("/scratch/kchern2/FIRESIM_RUNS_DIR/sim_slot_0")

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


class XnnpackTest:
    """Test wrapper for XNNPACK microkernels.

    Injects candidate function body between SUBSTITUTE markers in the test
    harness. Correctness is checked by the XNNPACK tester class, not inline.
    """
    def __init__(self, test_file: pathlib.Path):
        self.test_file = test_file

    def _substitute(self, code_str: str) -> str:
        """Inject code between SUBSTITUTE markers in the test harness."""
        content = self.test_file.read_text().splitlines()
        result = []
        substituting = False
        for line in content:
            if "// SUBSTITUTE HERE" in line:
                substituting = True
                result.append(line)
                result.append(code_str)
            elif "// SUBSTITUTE END" in line:
                substituting = False
                result.append(line)
            elif not substituting:
                result.append(line)
        return "\n".join(result)

    def get_test_code(self, code_str: str) -> str:
        """Inject candidate function body into the test harness."""
        return self._substitute(code_str)



MAX_BUILD_SLOTS = min(4,os.cpu_count())


def _build_and_run_spike(args: tuple) -> tuple:
    
    code_contents, slot_id, candidate_idx = args
    slot_dir = XNNPACK_TEMP_DIR / f"build_slot_{slot_id}"
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
                XNNPACK_ZEPHYR_APP_PATH, app_dir,
                ignore=shutil.ignore_patterns('build*', '__pycache__', '*.o', '*.elf')
            )

        (app_dir / "src" / "main.cpp").write_text(code_contents)

        pristine_flag = "-p" if is_first else ""
        build_cmd = f"""
            cd {XNNPACK_ZEPHYR_BASE} && \
            source scripts/set_envvars_sdk.sh && \
            source tools/miniforge3/etc/profile.d/conda.sh && \
            conda activate zephyr && \
            west build {pristine_flag} -b spike_riscv64 -d {build_dir} {app_dir} -DXNNPACK_ENABLE_RISCV_VECTOR=ON -DXNNPACK_ENABLE_RISCV_GEMMINI=OFF 
        """
        result = subprocess.run(
            ["bash", "-c", build_cmd],
            capture_output=True,
            timeout=XNNPACK_COMPILE_TIMEOUT
        )
        if result.returncode != 0:
            logger.info("[slot %d] candidate %d: build failed", slot_id, candidate_idx)
            return (candidate_idx, f"Compile error: {result.stderr.decode()}")

        binary = build_dir / "zephyr" / "zephyr.elf"
        if not binary.exists():
            return (candidate_idx, f"Binary not found at {binary}")

        logger.info("[slot %d] candidate %d: build done%s", slot_id, candidate_idx, " (pristine)" if is_first else "")

    except subprocess.TimeoutExpired:
        logger.info("[slot %d] candidate %d: build timed out", slot_id, candidate_idx)
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
            timeout=XNNPACK_SPIKE_TIMEOUT
        )
        logger.info("[slot %d] candidate %d: spike done", slot_id, candidate_idx)
        return (candidate_idx, result.stdout)
    except subprocess.TimeoutExpired:
        return (candidate_idx, "Timeout")
    except Exception as e:
        return (candidate_idx, f"Spike error: {str(e)}")


def _run_slot(args: tuple) -> list[tuple]:
    """Run all tasks for a single build slot sequentially."""
    slot_id, task_list = args
    return [_build_and_run_spike((code, slot_id, idx)) for code, idx in task_list]


def run_spike_mp(code_contents_lst: list[str]) -> list[str]:

    results = ["Error"] * len(code_contents_lst)

    num_slots = min(len(code_contents_lst), MAX_BUILD_SLOTS)
    logger.info("Building & running spike on %d candidates across %d slots...",
                len(code_contents_lst), num_slots)

    # Group tasks by slot so each slot is only accessed by one worker
    slot_tasks: list[list[tuple]] = [[] for _ in range(num_slots)]
    for i, code in enumerate(code_contents_lst):
        slot_tasks[i % num_slots].append((code, i))

    with multiprocessing.Pool(num_slots) as pool:
        for slot_results in pool.imap_unordered(_run_slot, enumerate(slot_tasks)):
            for candidate_idx, stdout in slot_results:
                results[candidate_idx] = stdout

    return results


FIRESIM_BUILD_SLOT = 0  # Reuse spike slot 0 -- spike finishes before firesim starts

def build_firesim_binary(code_contents: str) -> pathlib.Path | str:
   
    slot_dir = XNNPACK_TEMP_DIR / f"build_slot_{FIRESIM_BUILD_SLOT}"
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
                XNNPACK_ZEPHYR_APP_PATH, app_dir,
                ignore=shutil.ignore_patterns('build*', '__pycache__', '*.o', '*.elf')
            )

        (app_dir / "src" / "main.cpp").write_text(code_contents)

        pristine_flag = "-p" if is_first else ""
        build_cmd = f"""
            cd {XNNPACK_ZEPHYR_BASE} && \
            source scripts/set_envvars_sdk.sh && \
            source tools/miniforge3/etc/profile.d/conda.sh && \
            conda activate zephyr && \
            west build {pristine_flag} -b spike_riscv64 -d {build_dir} {app_dir} -DXNNPACK_ENABLE_RISCV_VECTOR=ON -DXNNPACK_ENABLE_RISCV_GEMMINI=OFF
        """
        result = subprocess.run(
            ["bash", "-c", build_cmd],
            capture_output=True,
            timeout=XNNPACK_COMPILE_TIMEOUT
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
                      n_expected: int,
                      candidate_order: list[int] = None,
                      per_candidate_timeout: float = XNNPACK_FIRESIM_PER_CANDIDATE_TIMEOUT,
                      min_timeout: float = XNNPACK_FIRESIM_MIN_TIMEOUT) -> tuple[dict[int, int], str | None]:
    """
    Run binary on FireSim with uartlog polling for hang detection.

    Monitors the live uartlog in sim_slot_0 for "ID X latency:" lines.
    If no new result appears within per_candidate_timeout, declares a hang,
    kills FireSim, and returns partial results collected so far.
    """
    # 1. Copy binary to workload directory
    workload_dir = firesim_path / "deploy" / "workloads" / "saturn"
    workload_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(binary_path, workload_dir / "saturn_test-baremetal")

    firesim_setup = f"cd {firesim_path} && source {firesim_path}/sourceme-manager.sh"

    # 2. Run infrasetup (blocking)
    logger.info("Running `firesim infrasetup`")
    infrasetup = subprocess.run(
        ["bash", "-c", f"{firesim_setup} && firesim infrasetup"],
        capture_output=True, text=True
    )
    if infrasetup.returncode != 0:
        logger.error("firesim infrasetup failed: %s", infrasetup.stderr or infrasetup.stdout)
        raise RuntimeError(f"firesim infrasetup failed with return code {infrasetup.returncode}")

    # 3. Clear old uartlog so we only see fresh output
    uartlog_path = XNNPACK_FIRESIM_SIM_SLOT_DIR / "uartlog"
    if uartlog_path.exists():
        uartlog_path.write_text("")

    # 4. Launch runworkload in background
    logger.info("Running `firesim runworkload` (polling uartlog for results)")
    proc = subprocess.Popen(
        ["bash", "-c", f"{firesim_setup} && firesim runworkload"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # 5. Poll uartlog for results
    pattern = re.compile(r"ID (\d+) latency: (\d+) cycles")
    results = {}
    last_result_time = time.time()
    poll_interval = 1.0
    timeout = min_timeout + per_candidate_timeout * n_expected
    hung_candidate = None

    try:
        while len(results) < n_expected:
            # Check if process exited
            if proc.poll() is not None:
                # Process finished — parse final uartlog
                if uartlog_path.exists():
                    for m in pattern.finditer(uartlog_path.read_text()):
                        results[int(m.group(1))] = int(m.group(2))
                break

            # Read current uartlog
            if uartlog_path.exists():
                content = uartlog_path.read_text()
                new_found = False
                for m in pattern.finditer(content):
                    idx, latency = int(m.group(1)), int(m.group(2))
                    if idx not in results:
                        results[idx] = latency
                        logger.info("FireSim: candidate %d = %d cycles", idx, latency)
                        new_found = True
                if new_found:
                    last_result_time = time.time()

            # Check for hang — kill if no new result in per_candidate_timeout
            # First candidate gets extra time for runworkload setup
            hang_timeout = per_candidate_timeout + (15 if not results else 0)
            elapsed = time.time() - last_result_time
            if elapsed > hang_timeout:
                # Determine which candidate hung (next expected after last seen)
                hung_candidate = _identify_hung_candidate(content if uartlog_path.exists() else "", results, candidate_order)
                logger.warning("FireSim: no new result for %.0fs, candidate %s appears to hang",
                               elapsed, hung_candidate)
                break

            time.sleep(poll_interval)

    finally:
        # Kill FireSim if still running
        if proc.poll() is None:
            logger.info("Killing FireSim (poll exit)")
            kill_cmd = f"{firesim_setup} && firesim kill"
            subprocess.run(["bash", "-c", kill_cmd],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            proc.wait(timeout=120)

    if hung_candidate is not None:
        logger.warning("FireSim: hung on candidate %s, returning %d partial results",
                       hung_candidate, len(results))
    else:
        logger.info("FireSim: collected %d/%d results", len(results), n_expected)

    return results, hung_candidate


def _identify_hung_candidate(uartlog_content: str, results: dict[int, int],
                             candidate_order: list[int] = None) -> str:
    """Try to identify which candidate hung based on uartlog content and run order."""
    if candidate_order:
        # Find the first candidate in run order that has no result
        for idx in candidate_order:
            if idx not in results:
                return str(idx)
    if not results:
        return "unknown (no results before hang)"
    # Fallback: the hung candidate is likely the next one after the last successful result
    last_id = max(results.keys())
    return str(last_id + 1)


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


class XnnpackEvalBackend(EvalBackend):
    """
    Hardware backend for evaluating RVV code on XNNPACK core using spike/FireSim.

    """

    def __init__(self,
                 vlen: int = 256,
                 elen: int = 64,
                 chipyard_path: str = None):

        super().__init__()
        self.vlen = vlen
        self.elen = elen

        # Set paths
        chipyard_path = chipyard_path or XNNPACK_CHIPYARD_PATH
        if chipyard_path:
            self.chipyard_path = pathlib.Path(chipyard_path).resolve()
            self.xnnpack_path = self.chipyard_path / "generators" / "firechip"
            self.firesim_path = self.chipyard_path / "sims" / "firesim"
        else:
            self.chipyard_path = None
            self.xnnpack_path = None
            self.firesim_path = None

    def __repr__(self):
        return f"XnnpackEvalBackend(vlen={self.vlen}, elen={self.elen})"

    def evaluate_code_spike(self, prob: Prob, code_strs: list[str]) -> List[dict]:
        """Convenience method for spike-only evaluation."""
        return self.evaluate_code(prob, code_strs, "spike")

    def evaluate_code_firesim(self, prob: Prob, code_strs: list[str]) -> List[dict]:
        """Convenience method for FireSim evaluation."""
        return self.evaluate_code(prob, code_strs, "firesim")

    def evaluate_code(self, prob, code_strs: list[str], simulator: str) -> List[dict]:
        """
        Evaluate code candidates and return results.

        """
        xnnpack_tests = [XnnpackTest(t.test_file) for t in prob.tests]

        if not self.xnnpack_path or not self.xnnpack_path.exists():
            raise ValueError(
                "XNNPACK path not configured. Set XNNPACK_CHIPYARD_PATH environment variable "
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
        for test_i, test in enumerate(xnnpack_tests):
            logger.info("Running spike on %d code candidates for test %d", len(code_strs), test_i)

            # Get test code with each candidate injected
            test_codes = [test.get_test_code(code_str) for code_str in clean_code_strs]

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
        Run passing candidates on FireSim with uartlog polling.

        Monitors the live uartlog for results. If a candidate hangs,
        kills FireSim, removes the hanging candidate, and re-runs
        the remaining untimed candidates. Repeats until all candidates
        are timed or all remaining ones hang.
        """
        xnnpack_tests = [XnnpackTest(t.test_file) for t in prob.tests]
        if not xnnpack_tests:
            raise ValueError("No tests available for FireSim template")
        first_test = xnnpack_tests[0]

        all_results = {}
        remaining_codes = list(passing_codes)
        remaining_indices = list(passing_indices)
        hung_indices = set()

        while remaining_indices:
            combined_code = self._build_firesim_combined_code(
                remaining_codes, remaining_indices, first_test, prob
            )
            binary_result = build_firesim_binary(combined_code)
            if isinstance(binary_result, str):
                raise RuntimeError(f"Failed to compile FireSim binary: {binary_result}")

            results, hung_candidate = run_firesim_batch(
                binary_result, self.firesim_path,
                n_expected=len(remaining_indices),
                candidate_order=remaining_indices
            )

            # Collect successful results
            all_results.update(results)

            if hung_candidate is None:
                # All candidates completed
                break

            # Identify the hung candidate index
            try:
                hung_idx = int(hung_candidate)
            except (ValueError, TypeError):
                logger.warning("Could not identify hung candidate (%s), stopping retries", hung_candidate)
                break

            hung_indices.add(hung_idx)
            logger.info("Removing hung candidate %d, re-running %d remaining",
                        hung_idx, len(remaining_indices) - len(results) - 1)

            # Rebuild remaining list: exclude already-timed and hung candidates
            timed_or_hung = set(all_results.keys()) | hung_indices
            new_codes = []
            new_indices = []
            for code, idx in zip(passing_codes, passing_indices):
                if idx not in timed_or_hung:
                    new_codes.append(code)
                    new_indices.append(idx)

            remaining_codes = new_codes
            remaining_indices = new_indices

            if not remaining_indices:
                logger.info("No more candidates to re-run after removing hung ones")

        return all_results

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
        Build combined C++ file with all passing candidates for FireSim.

        Each candidate is wrapped in its own noinline function to isolate
        vector register allocation. The file is generated from the test
        harness preamble (includes/utilities) plus a new main() that runs
        correctness checks via the XNNPACK tester and measures latency.

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

        # Extract preamble from test harness (includes + utility functions,
        # everything before the candidate_kernel definition)
        test_content = test.test_file.read_text()
        cut_points = ["static void candidate_kernel", "// SUBSTITUTE HERE"]
        cut_pos = len(test_content)
        for marker in cut_points:
            pos = test_content.find(marker)
            if pos != -1 and pos < cut_pos:
                cut_pos = pos
        preamble = test_content[:cut_pos]

        # Generate noinline wrapper functions for each candidate
        func_defs = []
        for code, orig_idx in zip(passing_codes, passing_indices):
            func_defs.append(
                f"__attribute__((noinline)) void run_candidate_{orig_idx}({param_sig}) {{\n"
                f"{code}\n"
                f"}}\n"
            )

        # Generate per-candidate blocks: timing via tester.kernel_cycles()
        # Correctness already verified on spike; FireSim is for accurate latency only.
        is_dwconv2d = "DWConv2DMicrokernelTester" in test_content

        candidate_blocks = []
        for orig_idx in passing_indices:
            if is_dwconv2d:
                candidate_blocks.append(f"""
    {{
        static const size_t test_sizes[][2] = {{
            {{7, 7}}, {{12, 8}}, {{14, 14}}, {{28, 28}}, {{256, 256}}
        }};
        DWConv2DMicrokernelTester tester;
        tester.kernel_height(3)
              .kernel_width(3)
              .subsampling(1)
              .padding_left(1)
              .padding_right(1)
              .padding_top(1)
              .padding_bottom(1)
              .iterations(1);
        for (size_t i = 0; i < sizeof(test_sizes) / sizeof(test_sizes[0]); i++) {{
            tester.input_height(test_sizes[i][0])
                  .input_width(test_sizes[i][1])
                  .Test(run_candidate_{orig_idx}, xnn_init_f32_minmax_scalar_params);
        }}
        printf("ID {orig_idx} latency: %lu cycles\\n", tester.kernel_cycles());
    }}""")
            else:
                candidate_blocks.append(f"""
    {{
        RAddStoreExpMinusMaxMicrokernelTester tester;
        for (size_t elems = BATCH_SIZE - 8; elems <= BATCH_SIZE + 8; elems++) {{
            tester.elements(elems)
                  .iterations(1)
                  .Test(run_candidate_{orig_idx}, nullptr);
        }}
        printf("ID {orig_idx} latency: %lu cycles\\n", tester.kernel_cycles());
    }}""")

        combined = preamble
        combined += "\n".join(func_defs)
        combined += f"""
int main() {{
    volatile double dummy = 1.0;
    dummy = dummy / 1.0000001;
{"".join(candidate_blocks)}

    sys_reboot(SYS_REBOOT_COLD);
}}
"""
        return combined


if __name__ == "__main__":
    prob = Prob("qs8", 0)
    files = [SOLS_DIR / "qs8" / "qs8-vaddc.c"]
    if files[0].exists():
        code_str = files[0].read_text()
        code_strs = [code_str, code_str, code_str]
        backend = XnnpackEvalBackend()
        stats = backend.evaluate_code(prob, code_strs, "firesim")
        for i, stat in enumerate(stats):
            print(f"  Candidate {i}: {stat}")
    else:
        print("No test files found. Create tests in tests/qs8/ and sols in sols/qs8/")
