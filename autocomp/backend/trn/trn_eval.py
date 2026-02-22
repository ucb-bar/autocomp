import os
import pathlib
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from datetime import datetime

from autocomp.common import logger, TESTS_DIR
from autocomp.search.prob import Prob
from autocomp.backend.eval_backend import EvalBackend

COMBINED_RESULTS_MARKER = "===COMBINED_RESULTS==="

# Path to the nrt_helper module shipped alongside this file
_NRT_HELPER_PATH = pathlib.Path(__file__).parent / "nrt_helper.py"


class TrnEvalBackend(EvalBackend):

    def __init__(self, parallel: bool = True):
        self.parallel = parallel

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _extract_latency(self, stdout: str) -> float:
        """Extract latency from stdout using pattern 'Latency: <latency> ms'"""
        for line in stdout.split('\n'):
            if 'Latency:' in line and 'ms' in line:
                parts = line.split('Latency:')[1].split('ms')[0].strip()
                try:
                    return float(parts)
                except ValueError:
                    continue
        return None

    def _extract_ref_func_name(self, test_code: str) -> str:
        """Extract the reference function name from the test_nki() call site."""
        for line in test_code.split('\n'):
            stripped = line.strip()
            if 'test_nki(' in stripped and not stripped.startswith('def '):
                match = re.search(r'test_nki\(\s*(\w+)\s*,', stripped)
                if match:
                    return match.group(1)
        return "ref"

    def _extract_imports(self, test_code: str) -> str:
        """Extract all import lines from the test code."""
        import_lines = []
        for line in test_code.split('\n'):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                import_lines.append(line)
        return '\n'.join(import_lines) + '\n' if import_lines else ''

    def _strip_main_block(self, code: str) -> str:
        """Remove the if __name__ == '__main__': block from code."""
        lines = code.split('\n')
        out_lines = []
        in_main_block = False
        for line in lines:
            if line.strip().startswith('if __name__'):
                in_main_block = True
                continue
            if in_main_block:
                if line.strip() == '' or line.startswith(' ') or line.startswith('\t'):
                    continue
                else:
                    in_main_block = False
            if not in_main_block:
                out_lines.append(line)
        return '\n'.join(out_lines)

    def _parse_combined_results(self, stdout: str, num_candidates: int) -> list[dict] | None:
        """Parse structured JSON results from a combined evaluation script."""
        for line in stdout.split('\n'):
            if line.startswith(COMBINED_RESULTS_MARKER):
                try:
                    results = json.loads(line[len(COMBINED_RESULTS_MARKER):])
                    if len(results) == num_candidates:
                        return results
                    logger.error(f"Combined results count mismatch: "
                                 f"got {len(results)}, expected {num_candidates}")
                    return None
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse combined results JSON: {e}")
                    return None
        return None

    # ------------------------------------------------------------------ #
    #  Candidate file management                                         #
    # ------------------------------------------------------------------ #

    def _write_candidate_files(self, test_code: str, code_strs: list[str],
                               temp_dir: pathlib.Path) -> None:
        """Write each candidate to its own .py file for importlib."""
        imports_header = self._extract_imports(test_code)
        candidates_dir = temp_dir / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        for i, code_str in enumerate(code_strs):
            with open(candidates_dir / f"candidate_{i}.py", "w") as f:
                f.write(imports_header + "\n" + code_str)
        with open(candidates_dir / "__init__.py", "w") as f:
            f.write("")

    # ------------------------------------------------------------------ #
    #  Phase 1: Parallel NEFF compilation                                #
    # ------------------------------------------------------------------ #

    def _generate_compile_script(self, test_code: str, idx: int,
                                 temp_dir: pathlib.Path,
                                 neff_dir: pathlib.Path,
                                 is_ref: bool = False) -> str:
        """Generate a Python script that compiles a kernel to a NEFF.

        If *is_ref* is True, compiles the reference kernel instead of a
        candidate.  Strategy: wrap the @nki.jit function with
        nki.baremetal(save_neff_name=...) and call test_nki(wrapped, wrapped)
        to trigger compilation with the correct input shapes.  The NEFF is
        saved to disk during compilation (before execution), so NeuronCore
        contention errors during execution are harmless.
        """
        preamble = test_code.split("# SUBSTITUTE HERE")[0]
        postamble_raw = (test_code.split("# SUBSTITUTE HERE")[1]
                         if "# SUBSTITUTE HERE" in test_code else "")
        postamble = self._strip_main_block(postamble_raw)
        ref_func_name = self._extract_ref_func_name(test_code)

        if is_ref:
            neff_path = str((neff_dir / "ref.neff").resolve())
            func_setup = f'''\
_target_func = {ref_func_name}
'''
        else:
            neff_path = str(
                (neff_dir / f"candidate_{idx}.neff").resolve())
            func_setup = f'''\
import importlib
_mod = importlib.import_module("candidates.candidate_{idx}")
_target_func = getattr(_mod, "test", None)
if _target_func is None:
    print(json.dumps({{"compiled": False, "error": "No test function defined"}}))
    sys.exit(0)
'''

        return f'''\
import sys, json, os, traceback
sys.path.insert(0, {repr(str(temp_dir.resolve()))})

import neuronxcc.nki as nki

# === Test preamble ===
{preamble}
# === Test postamble ===
{postamble}

_neff_path = {repr(neff_path)}
{func_setup}
# Wrap with nki.baremetal to save NEFF during compilation
_raw = _target_func.func if hasattr(_target_func, "func") else _target_func
_bm = nki.baremetal(_raw, save_neff_name=_neff_path)

# Trigger compilation by calling test_nki with the baremetal wrapper as
# both ref and candidate.  The first call compiles and saves the NEFF;
# execution may fail (NeuronCore contention) but the NEFF is already on disk.
_error_msg = ""
try:
    test_nki(_bm, _bm)
except Exception:
    # If the NEFF was saved, execution failed (e.g. NeuronCore contention)
    # which is fine.  If not, compilation itself failed — capture the error.
    if not os.path.exists(_neff_path):
        _error_msg = traceback.format_exc()

print(json.dumps({{"compiled": os.path.exists(_neff_path), "error": _error_msg}}))
'''

    def _compile_candidates_parallel(self, test_code: str,
                                     code_strs: list[str],
                                     temp_dir: pathlib.Path):
        """Phase 1: compile ref + all candidates in parallel, save NEFFs.

        Returns (compiled_dict, neff_dir) where compiled_dict maps
        candidate index → bool indicating whether the NEFF was saved.
        The ref NEFF is always compiled (needed for Phase 2 correctness).
        """
        neff_dir = temp_dir / "neffs"
        neff_dir.mkdir(parents=True, exist_ok=True)

        # Generate compile scripts: ref + all candidates
        scripts = {}  # label -> path

        ref_script = self._generate_compile_script(
            test_code, -1, temp_dir, neff_dir, is_ref=True)
        ref_path = temp_dir / "compile_ref.py"
        with open(ref_path, "w") as f:
            f.write(ref_script)
        scripts["ref"] = ref_path

        for i in range(len(code_strs)):
            script = self._generate_compile_script(
                test_code, i, temp_dir, neff_dir)
            path = temp_dir / f"compile_{i}.py"
            with open(path, "w") as f:
                f.write(script)
            scripts[i] = path

        def run_compile(label):
            cmd = ["python", str(scripts[label].resolve())]
            out_name = f"compile_{label}_output.txt"
            try:
                p = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=600)
                with open(temp_dir / out_name, "w") as f:
                    f.write(f"=== STDOUT ===\n{p.stdout}\n"
                            f"=== STDERR ===\n{p.stderr}")
                if label == "ref":
                    ok = (neff_dir / "ref.neff").exists()
                else:
                    ok = (neff_dir / f"candidate_{label}.neff").exists()

                # Extract error from the compile script's JSON output
                error_msg = ""
                if not ok:
                    # Try to parse the JSON line from stdout
                    for line in p.stdout.strip().splitlines():
                        try:
                            data = json.loads(line)
                            if data.get("error"):
                                error_msg = data["error"]
                                break
                        except (json.JSONDecodeError, TypeError):
                            continue
                    # Fall back to stderr if no JSON error was found
                    if not error_msg and p.stderr:
                        error_msg = p.stderr
                    # Last resort: non-zero exit with no other info
                    if not error_msg and p.returncode != 0:
                        error_msg = (f"Compile script exited with code "
                                     f"{p.returncode}")

                return label, ok, error_msg
            except subprocess.TimeoutExpired:
                logger.error(f"Compile {label} timed out after 600s")
                return label, False, "Timed out after 600 seconds"
            except Exception as e:
                logger.error(f"Compile {label} error: {e}")
                return label, False, str(e)

        compiled = {}        # label -> bool
        compile_errors = {}  # label -> error_msg (only for failures)
        all_labels = ["ref"] + list(range(len(code_strs)))
        max_parallel = min(os.cpu_count() or 1, len(all_labels))
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {executor.submit(run_compile, label): label
                       for label in all_labels}
            for future in as_completed(futures):
                label, ok, error_msg = future.result()
                compiled[label] = ok
                if not ok:
                    compile_errors[label] = error_msg
                logger.info(f"Compile {label}: {'OK' if ok else 'FAILED'}")

        return compiled, compile_errors, neff_dir

    # ------------------------------------------------------------------ #
    #  Phase 2: Sequential correctness + benchmark via libnrt            #
    # ------------------------------------------------------------------ #

    def _generate_phase2_script(self, test_code: str, num_candidates: int,
                                temp_dir: pathlib.Path,
                                neff_dir: pathlib.Path,
                                compiled: dict,
                                compile_errors: dict) -> str:
        """Generate Phase 2 script using *only* libnrt (no NKI).

        The script:
        1. Loads the ref NEFF via libnrt (instant, no compilation).
        2. Generates random inputs matching the NEFF's tensor metadata.
        3. Executes the ref NEFF to get reference outputs.
        4. For each candidate, loads its NEFF, executes with the same
           inputs, and compares outputs for correctness.
        5. For passing candidates, benchmarks via nrt_execute loop.

        Only one model is loaded at a time (sequential load → execute →
        cleanup) to avoid NeuronCore contention.
        """
        ref_neff = str((neff_dir / "ref.neff").resolve())
        neff_paths = {}
        candidate_compile_errors = {}
        for i in range(num_candidates):
            if compiled.get(i, False):
                neff_paths[i] = str(
                    (neff_dir / f"candidate_{i}.neff").resolve())
            else:
                candidate_compile_errors[i] = compile_errors.get(
                    i, "NEFF compilation failed in Phase 1")

        return f'''\
import sys, json, os, traceback
import numpy as np
sys.path.insert(0, {repr(str(temp_dir.resolve()))})

from nrt_helper import NrtModel, nrt_init, nrt_close

_ref_neff = {repr(ref_neff)}
_neff_paths = {repr(neff_paths)}
_compile_errors = {repr(candidate_compile_errors)}
_num_candidates = {num_candidates}
_all_results = [None] * _num_candidates
_NUM_CORRECTNESS_ROUNDS = 2

nrt_init()

# --- Step 1: execute ref NEFF to get reference outputs ---
_ref_model = NrtModel(_ref_neff)

# Generate random inputs from the NEFF tensor metadata (shape + dtype)
_ref_outputs_per_round = []
_inputs_per_round = []
for _round in range(_NUM_CORRECTNESS_ROUNDS):
    _inputs = []
    for _name, _size, _dtype, _shape in _ref_model._inputs_info:
        _inputs.append(np.random.rand(*_shape).astype(_dtype))
    _ref_out = _ref_model(*_inputs)
    _inputs_per_round.append(_inputs)
    _ref_outputs_per_round.append(_ref_out)

_ref_model.cleanup()

# --- Step 2: evaluate each candidate ---
for _idx in range(_num_candidates):
    _result = {{"correct": False, "latency": None, "stdout": "", "stderr": ""}}

    if _idx not in _neff_paths:
        _result["stderr"] = _compile_errors.get(_idx, "NEFF compilation failed in Phase 1")
        _all_results[_idx] = _result
        continue

    _cand_model = None
    try:
        _cand_model = NrtModel(_neff_paths[_idx])

        # Correctness: run with same inputs, compare outputs
        _passed = True
        for _round in range(_NUM_CORRECTNESS_ROUNDS):
            _cand_out = _cand_model(*_inputs_per_round[_round])
            _ref_out = _ref_outputs_per_round[_round]

            # Normalise to list of arrays
            _ro = [_ref_out] if isinstance(_ref_out, np.ndarray) else list(_ref_out)
            _co = [_cand_out] if isinstance(_cand_out, np.ndarray) else list(_cand_out)

            for _r, _c in zip(_ro, _co):
                if not np.allclose(
                    _r.astype(np.float32), _c.astype(np.float32),
                    atol=1, rtol=1e-2,
                ):
                    _passed = False
                    break
            if not _passed:
                break

        if not _passed:
            _result["stderr"] = "Test failed: output mismatch"
            _all_results[_idx] = _result
            continue

        # Benchmark via nrt_execute loop
        _latency = _cand_model.benchmark(warmup=10, iters=100)

        _result["correct"] = True
        _result["latency"] = round(_latency, 3)
        _result["stdout"] = "Latency: {{:.3f}} ms (P99)\\n".format(_latency)

    except Exception:
        _result["stderr"] = traceback.format_exc()
    finally:
        if _cand_model is not None:
            try:
                _cand_model.cleanup()
            except Exception:
                pass

    _all_results[_idx] = _result

nrt_close()
print("{COMBINED_RESULTS_MARKER}" + json.dumps(_all_results))
'''

    # ------------------------------------------------------------------ #
    #  Fallback: single-candidate evaluation                             #
    # ------------------------------------------------------------------ #

    def _evaluate_single(self, test_code: str, code_str: str,
                         temp_dir: pathlib.Path, idx: int) -> dict:
        """Evaluate a single candidate in its own subprocess (fallback)."""
        test_code_i = test_code.replace("# SUBSTITUTE HERE", code_str)
        with open(temp_dir / f"code_{idx}.py", "w") as f:
            f.write(test_code_i)

        cmd = ["python", str(temp_dir.resolve() / f"code_{idx}.py")]
        logger.info(f"Running command {' '.join(cmd)}")
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=300)
        except subprocess.TimeoutExpired:
            logger.error(f"Code {idx} timed out after 300 seconds")
            return {"correct": False, "latency": None, "stdout": "",
                    "stderr": "Timed out after 300 seconds"}

        with open(temp_dir / f"code_{idx}_output.txt", "w") as f:
            f.write("=== STDOUT ===\n")
            f.write(p.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(p.stderr)

        result_dict = {
            "correct": False,
            "latency": None,
            "stdout": p.stdout,
            "stderr": p.stderr,
        }

        if p.returncode != 0:
            logger.error(f"Code {idx} failed to run")
            return result_dict

        latency = self._extract_latency(p.stdout)
        if latency is None:
            logger.error(f"Code {idx} did not produce latency output")
            return result_dict

        logger.info(f"Code {idx} latency: {latency}")
        result_dict["correct"] = True
        result_dict["latency"] = latency
        return result_dict

    # ------------------------------------------------------------------ #
    #  Orchestration                                                     #
    # ------------------------------------------------------------------ #

    def _try_combined_evaluation(self, test_code: str, code_strs: list[str],
                                 temp_dir: pathlib.Path) -> list[dict] | None:
        """Two-phase evaluation: parallel compile then sequential eval.

        Phase 1 – Compile all candidates in parallel via nki.baremetal,
                  saving NEFFs to disk.
        Phase 2 – Load each NEFF via libnrt (instant, no recompilation),
                  run test_nki for correctness, and nrt_execute loop for
                  latency benchmarking.

        Returns None on infrastructure failure (caller falls back to
        per-candidate subprocess evaluation).
        """
        # Write candidate .py files
        self._write_candidate_files(test_code, code_strs, temp_dir)

        # Copy nrt_helper.py into temp_dir so Phase 2 script can import it
        shutil.copy2(_NRT_HELPER_PATH, temp_dir / "nrt_helper.py")

        # Phase 1: parallel compilation (ref + all candidates)
        logger.info(f"Phase 1: compiling ref + {len(code_strs)} candidates "
                     "in parallel")
        compiled, compile_errors, neff_dir = \
            self._compile_candidates_parallel(
                test_code, code_strs, temp_dir)
        compiled_count = sum(1 for k, v in compiled.items()
                             if v and k != "ref")
        logger.info(f"Phase 1 complete: ref={'OK' if compiled.get('ref') else 'FAILED'}, "
                     f"{compiled_count}/{len(code_strs)} candidate NEFFs saved")

        if not compiled.get("ref"):
            logger.error("Ref NEFF compilation failed, cannot run Phase 2")
            return None

        # Phase 2: sequential correctness + benchmark
        logger.info("Phase 2: correctness + benchmark via libnrt")
        phase2_script = self._generate_phase2_script(
            test_code, len(code_strs), temp_dir, neff_dir,
            compiled, compile_errors)
        phase2_path = temp_dir / "phase2_eval.py"
        with open(phase2_path, "w") as f:
            f.write(phase2_script)

        # ref compilation (~60s) + per-candidate libnrt eval (~10s each)
        timeout = 120 + 60 * len(code_strs)
        cmd = ["python", str(phase2_path.resolve())]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.error(f"Phase 2 timed out after {timeout}s")
            return None

        with open(temp_dir / "phase2_output.txt", "w") as f:
            f.write(f"=== STDOUT ===\n{p.stdout}\n"
                    f"=== STDERR ===\n{p.stderr}")

        results = self._parse_combined_results(p.stdout, len(code_strs))
        if results is None:
            logger.error(f"Phase 2 did not produce valid results. "
                         f"returncode={p.returncode}, "
                         f"stderr={p.stderr[:500]}")
            return None

        for i, result in enumerate(results):
            if result["correct"]:
                logger.info(f"Code {i} latency: {result['latency']}")
            else:
                logger.error(f"Code {i} failed")

        return results

    def evaluate_code(self, prob: Prob, code_strs: list[str],
                      simulator: str) -> List[dict]:
        """Evaluate candidates using parallel NEFF compilation + libnrt.

        Phase 1 compiles all candidates in parallel (CPU-bound), saving
        NEFFs to disk.  Phase 2 loads each NEFF via libnrt (no
        recompilation) for correctness and latency benchmarking.

        Falls back to individual subprocess evaluation on infrastructure
        errors.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = (pathlib.Path(__file__).parent / "tmp_files" /
                    "trn_eval" / timestamp)
        temp_dir.mkdir(parents=True, exist_ok=True)

        if prob.test_file:
            test_file = prob.test_file
        else:
            test_dir = TESTS_DIR / prob.prob_type
            matches = list(test_dir.glob(f"{prob.prob_id}_*.py"))
            if not matches:
                raise FileNotFoundError(
                    f"No test file found for {prob.prob_type} {prob.prob_id} "
                    f"in {test_dir}")
            test_file = matches[0]
        test_code = test_file.read_text()

        results = None
        if self.parallel:
            results = self._try_combined_evaluation(test_code, code_strs,
                                                    temp_dir)
        if results is not None:
            return results

        if self.parallel:
            logger.warning("Combined evaluation failed, falling back to "
                           "individual evaluation")
        results = []
        for i, code_str in enumerate(code_strs):
            results.append(self._evaluate_single(test_code, code_str,
                                                 temp_dir, i))
        return results
