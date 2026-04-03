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


def _test_is_nki_v1(test_code: str) -> bool:
    """Detect if a test file uses NKI v1 (neuronxcc.nki / baremetal) style.

    NKI v1 tests: ``import neuronxcc.nki as nki`` and pass numpy arrays.
    NKI v2 tests: ``import nki`` and use torch/XLA tensors.
    """
    for line in test_code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import neuronxcc.nki") or stripped.startswith("from neuronxcc.nki"):
            return True
    return False


def _ensure_platform_override():
    """Auto-set NEURON_PLATFORM_TARGET_OVERRIDE from EC2 IMDS if not already set."""
    if os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE"):
        return
    try:
        import urllib.request
        tok_req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "30"},
        )
        token = urllib.request.urlopen(tok_req, timeout=2).read().decode().strip()
        req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-type",
            headers={"X-aws-ec2-metadata-token": token},
        )
        instance_type = urllib.request.urlopen(req, timeout=2).read().decode().strip()
        platform = instance_type.split(".")[0]
        os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = platform
        logger.info(f"Auto-detected platform={platform} from instance={instance_type}")
    except Exception:
        logger.warning("Could not auto-detect NEURON_PLATFORM_TARGET_OVERRIDE from IMDS")


# Template for the __main__ block injected into NKI v2 test scripts.
# Placeholder __REF_FUNC__ is replaced at runtime.
_NKI_V2_MAIN_TEMPLATE = '''
if __name__ == "__main__":
    import os
    import time
    import hashlib
    import torch
    import torch_xla
    from torch_xla.core import xla_model as xm

    import json as _json
    from nki.compiler.backends.neuron.TraceKernel import TraceKernel
    from nki._version import _version_identifier
    _orig_specialize = TraceKernel.specialize_and_call
    _trace_cache = {}

    def _patched_specialize(self, boundargs, output_path_prefix=None):
        import tempfile as _tf, inspect as _inspect
        fn = getattr(self.func, "__name__", "kernel")
        sk = "".join(str(a.shape) + "_" for a in boundargs.args if hasattr(a, "shape"))
        try:
            src = _inspect.getsource(self.func)
        except (OSError, TypeError):
            src = ""
        cache_key = hashlib.md5(f"{fn}_{sk}_{src}".encode()).hexdigest()[:12]

        if cache_key in _trace_cache:
            cached = _trace_cache[cache_key]
            self.klir_binary = cached["klir_binary"]
            return cached["result"]

        klir_dir = os.path.join(_tf.gettempdir(), "klir_binaries", f"{fn}_{cache_key}")
        klir_file = os.path.join(klir_dir, f"{fn}.klir")
        meta_file = os.path.join(klir_dir, f"{fn}_metadata.json")

        if os.path.exists(klir_file) and os.path.exists(meta_file):
            with open(meta_file, "r") as mf:
                metadata = _json.load(mf)
            inputs = metadata.get("inputs", [])
            outputs = metadata.get("outputs", [])
            shared = metadata.get("sharedConstants", [])
            klir_binary = {
                "binary": klir_file,
                "input_names": [i["name"] for i in inputs] + [c["name"] for c in shared],
                "output_names": [o["name"] for o in outputs],
                "version_identifier": _version_identifier,
            }
            self.klir_binary = klir_binary
            result = (0, klir_file, metadata)
            _trace_cache[cache_key] = {"klir_binary": klir_binary, "result": result}
            return result

        result = _orig_specialize(self, boundargs, output_path_prefix=cache_key)
        _, klir_path, metadata = result

        os.makedirs(klir_dir, exist_ok=True)
        with open(meta_file, "w") as mf:
            _json.dump(metadata, mf)

        _trace_cache[cache_key] = {"klir_binary": self.klir_binary, "result": result}
        return result

    TraceKernel.specialize_and_call = _patched_specialize

    test_result = test_nki(__REF_FUNC__, test)
    if not test_result:
        print("Test failed")
        exit(1)
    else:
        print("Test passed")

    class _LatencyResult:
        def __init__(self, times_us):
            self._times = sorted(times_us)
        def get_latency_percentile(self, pct):
            idx = int(len(self._times) * pct / 100.0)
            idx = min(idx, len(self._times) - 1)
            return self._times[idx]

    class _BenchmarkResult:
        def __init__(self, times_us):
            self.nc_latency = _LatencyResult(times_us)

    class _BenchmarkWrapper:
        def __init__(self, func, warmup, iters):
            self._func = func
            self._warmup = warmup
            self._iters = iters
            self.benchmark_result = None

        def __call__(self, *args, **kwargs):
            device = torch_xla.device()
            for _w in range(max(self._warmup, 4)):
                _out = self._func(*args, **kwargs)
                xm.mark_step()
            xm.wait_device_ops()

            _num_iters = max(self._iters, 10)
            times_us = []
            for _i in range(_num_iters):
                xm.wait_device_ops()
                _t0 = time.perf_counter()
                _out = self._func(*args, **kwargs)
                xm.mark_step()
                xm.wait_device_ops()
                _t1 = time.perf_counter()
                times_us.append((_t1 - _t0) * 1e6)

            self.benchmark_result = _BenchmarkResult(times_us)
            median_us = sorted(times_us)[len(times_us)//2]
            p99_us = sorted(times_us)[int(len(times_us)*0.99)]
            print(f"Latency: {median_us/1000.0:.3f} ms (median)")
            print(f"Latency: {p99_us/1000.0:.3f} ms (P99)")

    def _mock_benchmark(warmup=2, iters=10):
        def _decorator(func):
            return _BenchmarkWrapper(func, warmup, iters)
        return _decorator

    nki.benchmark = _mock_benchmark

    try:
        benchmark_nki(test)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Test failed")
        print(f"Benchmark error: {e}", flush=True)
        exit(1)
'''


class TrnEvalBackend(EvalBackend):
    def __init__(self, parallel: bool = True):
        _ensure_platform_override()
        self.parallel = parallel

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_num_cores() -> int:
        """Return the number of NeuronCores visible on this instance."""
        try:
            out = subprocess.check_output(
                ["neuron-ls", "--json-output"], text=True, timeout=5, stderr=subprocess.DEVNULL
            )
            data = json.loads(out)
            return sum(d.get("nc_count", 0) for d in data)
        except Exception:
            pass
        try:
            return len([f for f in os.listdir("/dev") if f.startswith("neuron")])
        except Exception:
            return 1

    def _extract_latency(self, stdout: str) -> float:
        """Extract latency from stdout using pattern 'Latency: <latency> ms'"""
        for line in stdout.split("\n"):
            if "Latency:" in line and "ms" in line:
                parts = line.split("Latency:")[1].split("ms")[0].strip()
                try:
                    return round(float(parts), 3)
                except ValueError:
                    continue
        return None

    def _extract_ref_func_name(self, test_code: str) -> str:
        """Extract the reference function name from the test_nki() call site."""
        for line in test_code.split("\n"):
            stripped = line.strip()
            if "test_nki(" in stripped and not stripped.startswith("def "):
                match = re.search(r"test_nki\(\s*(\w+)\s*,", stripped)
                if match:
                    return match.group(1)
        return "ref"

    def _extract_imports(self, test_code: str) -> str:
        """Extract all import lines from the test code."""
        import_lines = []
        for line in test_code.split("\n"):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                import_lines.append(line)
        return "\n".join(import_lines) + "\n" if import_lines else ""

    def _strip_main_block(self, code: str) -> str:
        """Remove the if __name__ == '__main__': block from code."""
        lines = code.split("\n")
        out_lines = []
        in_main_block = False
        for line in lines:
            if line.strip().startswith("if __name__"):
                in_main_block = True
                continue
            if in_main_block:
                if line.strip() == "" or line.startswith(" ") or line.startswith("\t"):
                    continue
                else:
                    in_main_block = False
            if not in_main_block:
                out_lines.append(line)
        return "\n".join(out_lines)

    def _parse_combined_results(self, stdout: str, num_impls: int) -> list[dict] | None:
        """Parse structured JSON results from a combined evaluation script."""
        for line in stdout.split("\n"):
            if line.startswith(COMBINED_RESULTS_MARKER):
                try:
                    results = json.loads(line[len(COMBINED_RESULTS_MARKER) :])
                    if len(results) == num_impls:
                        return results
                    logger.error(
                        f"Combined results count mismatch: "
                        f"got {len(results)}, expected {num_impls}"
                    )
                    return None
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse combined results JSON: {e}")
                    return None
        return None

    # ------------------------------------------------------------------ #
    #  Implementation file management                                     #
    # ------------------------------------------------------------------ #

    def _write_impl_files(
        self, test_code: str, code_strs: list[str], temp_dir: pathlib.Path
    ) -> None:
        """Write each implementation to its own .py file for importlib."""
        imports_header = self._extract_imports(test_code)
        impls_dir = temp_dir / "impls"
        impls_dir.mkdir(parents=True, exist_ok=True)
        for i, code_str in enumerate(code_strs):
            with open(impls_dir / f"impl_{i}.py", "w") as f:
                f.write(imports_header + "\n" + code_str)
        with open(impls_dir / "__init__.py", "w") as f:
            f.write("")

    # ------------------------------------------------------------------ #
    #  Phase 1: Parallel NEFF compilation                                #
    # ------------------------------------------------------------------ #

    def _generate_compile_script(
        self,
        test_code: str,
        idx: int,
        temp_dir: pathlib.Path,
        neff_dir: pathlib.Path,
        is_ref: bool = False,
    ) -> str:
        """Generate a Python script that compiles a kernel to a NEFF.

        NKI v1: uses nki.baremetal(save_neff_name=...) to save the NEFF directly.
        NKI v2: triggers compilation via @nki.jit with a deterministic cache
                prefix, then discovers the cached NEFF path from Neuron runtime
                logs and copies it to neff_dir.
        """
        preamble = test_code.split("# SUBSTITUTE HERE")[0]
        postamble_raw = (
            test_code.split("# SUBSTITUTE HERE")[1]
            if "# SUBSTITUTE HERE" in test_code
            else ""
        )
        postamble = self._strip_main_block(postamble_raw)
        ref_func_name = self._extract_ref_func_name(test_code)

        if is_ref:
            neff_path = str((neff_dir / "ref.neff").resolve())
            func_setup = f"""\
_target_func = {ref_func_name}
"""
        else:
            neff_path = str((neff_dir / f"impl_{idx}.neff").resolve())
            func_setup = f"""\
import importlib
_mod = importlib.import_module("impls.impl_{idx}")
_target_func = getattr(_mod, "test", None)
if _target_func is None:
    print(json.dumps({{"compiled": False, "error": "No test function defined"}}))
    sys.exit(0)
"""

        # 2-phase compilation only used for NKI v1 (baremetal).
        return self._compile_script_baremetal(
            preamble, postamble, func_setup, neff_path, temp_dir,
        )

    def _compile_script_baremetal(
        self, preamble, postamble, func_setup, neff_path, temp_dir,
    ) -> str:
        """NKI v1 compile script: use nki.baremetal to save NEFF directly."""
        return f"""\
import sys, json, os, traceback
sys.path.insert(0, {repr(str(temp_dir.resolve()))})

import neuronxcc.nki as nki

# === Test preamble ===
{preamble}
# === Test postamble ===
{postamble}

_neff_path = {repr(neff_path)}
{func_setup}
_raw = _target_func.func if hasattr(_target_func, "func") else _target_func
_bm = nki.baremetal(_raw, save_neff_name=_neff_path)

_error_msg = ""
try:
    test_nki(_bm, _bm)
except Exception:
    if not os.path.exists(_neff_path):
        _error_msg = traceback.format_exc()

print(json.dumps({{"compiled": os.path.exists(_neff_path), "error": _error_msg}}))
"""

    def _compile_impls_parallel(
        self, test_code: str, code_strs: list[str], temp_dir: pathlib.Path
    ):
        """Phase 1: compile ref + all implementations in parallel, save NEFFs.

        Returns (compiled_dict, neff_dir) where compiled_dict maps
        implementation index -> bool indicating whether the NEFF was saved.
        The ref NEFF is always compiled (needed for Phase 2 correctness).
        """
        neff_dir = temp_dir / "neffs"
        neff_dir.mkdir(parents=True, exist_ok=True)

        # Generate compile scripts: ref + all implementations
        scripts = {}  # label -> path

        ref_script = self._generate_compile_script(
            test_code, -1, temp_dir, neff_dir, is_ref=True
        )
        ref_path = temp_dir / "compile_ref.py"
        with open(ref_path, "w") as f:
            f.write(ref_script)
        scripts["ref"] = ref_path

        for i in range(len(code_strs)):
            script = self._generate_compile_script(test_code, i, temp_dir, neff_dir)
            path = temp_dir / f"compile_{i}.py"
            with open(path, "w") as f:
                f.write(script)
            scripts[i] = path

        def run_compile(label):
            cmd = ["python", str(scripts[label].resolve())]
            out_name = f"compile_{label}_output.txt"
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
                with open(temp_dir / out_name, "w") as f:
                    f.write(f"=== STDOUT ===\n{p.stdout}\n=== STDERR ===\n{p.stderr}")
                if label == "ref":
                    ok = (neff_dir / "ref.neff").exists()
                else:
                    ok = (neff_dir / f"impl_{label}.neff").exists()

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
                        error_msg = f"Compile script exited with code {p.returncode}"

                return label, ok, error_msg
            except subprocess.TimeoutExpired:
                logger.error(f"Compile {label} timed out after 600s")
                return label, False, "Timed out after 600 seconds"
            except Exception as e:
                logger.error(f"Compile {label} error: {e}")
                return label, False, str(e)

        compiled = {}  # label -> bool
        compile_errors = {}  # label -> error_msg (only for failures)
        all_labels = ["ref"] + list(range(len(code_strs)))

        max_parallel = min(os.cpu_count() or 1, len(all_labels))
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {
                executor.submit(run_compile, label): label
                for label in all_labels
            }
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

    def _generate_phase2_script(self, test_code: str, num_impls: int,
                                temp_dir: pathlib.Path,
                                neff_dir: pathlib.Path,
                                compiled: dict,
                                compile_errors: dict) -> str:
        """Generate Phase 2 script using *only* libnrt (no NKI).

        The script:
        1. Loads the ref NEFF via libnrt (instant, no compilation).
        2. Generates random inputs matching the NEFF's tensor metadata.
        3. Executes the ref NEFF to get reference outputs.
        4. For each implementation, loads its NEFF, executes with the same
           inputs, and compares outputs for correctness.
        5. For passing implementations, benchmarks via nrt_execute loop.

        Only one model is loaded at a time (sequential load -> execute ->
        cleanup) to avoid NeuronCore contention.
        """
        ref_neff = str((neff_dir / "ref.neff").resolve())
        neff_paths = {}
        impl_compile_errors = {}
        for i in range(num_impls):
            if compiled.get(i, False):
                neff_paths[i] = str(
                    (neff_dir / f"impl_{i}.neff").resolve())
            else:
                impl_compile_errors[i] = compile_errors.get(
                    i, "NEFF compilation failed in Phase 1")

        return f'''\
import sys, json, os, traceback
import numpy as np
sys.path.insert(0, {repr(str(temp_dir.resolve()))})

from nrt_helper import NrtModel, nrt_init, nrt_close

_ref_neff = {repr(ref_neff)}
_neff_paths = {repr(neff_paths)}
_compile_errors = {repr(impl_compile_errors)}
_num_impls = {num_impls}
_all_results = [None] * _num_impls
_NUM_CORRECTNESS_ROUNDS = 3

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

# --- Step 2: evaluate each implementation ---
for _idx in range(_num_impls):
    _result = {{"correct": False, "latency": None, "stdout": "", "stderr": ""}}

    if _idx not in _neff_paths:
        _result["stderr"] = _compile_errors.get(_idx, "NEFF compilation failed in Phase 1")
        _all_results[_idx] = _result
        continue

    _impl_model = None
    try:
        _impl_model = NrtModel(_neff_paths[_idx])

        # Correctness: run with same inputs, compare outputs
        _passed = True
        for _round in range(_NUM_CORRECTNESS_ROUNDS):
            _impl_out = _impl_model(*_inputs_per_round[_round])
            _ref_out = _ref_outputs_per_round[_round]

            # Normalise to list of arrays
            _ro = [_ref_out] if isinstance(_ref_out, np.ndarray) else list(_ref_out)
            _co = [_impl_out] if isinstance(_impl_out, np.ndarray) else list(_impl_out)

            for _r, _c in zip(_ro, _co):
                if not np.allclose(
                    _r.astype(np.float32), _c.astype(np.float32),
                    atol=1e-3, rtol=1e-3,
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
        _latency = _impl_model.benchmark(warmup=10, iters=100)

        _result["correct"] = True
        _result["latency"] = round(_latency, 3)
        _result["stdout"] = "Latency: {{:.3f}} ms (P99)\\n".format(_latency)

    except Exception:
        _result["stderr"] = traceback.format_exc()
    finally:
        if _impl_model is not None:
            try:
                _impl_model.cleanup()
            except Exception:
                pass

    _all_results[_idx] = _result

nrt_close()
print("{COMBINED_RESULTS_MARKER}" + json.dumps(_all_results))
'''

    # ------------------------------------------------------------------ #
    #  Fallback: single-implementation evaluation                         #
    # ------------------------------------------------------------------ #

    def _evaluate_single(
        self, test_code: str, code_str: str, temp_dir: pathlib.Path, idx: int,
        core_id: int = None,
    ) -> dict:
        """Evaluate a single implementation in its own subprocess (fallback).

        For NKI v2 tests, patches the __main__ block to use torch-based
        timing instead of nki.benchmark. For NKI v1 tests, runs as-is.
        """
        test_code_i = test_code.replace("# SUBSTITUTE HERE", code_str)
        ref_func_name = self._extract_ref_func_name(test_code)

        # NKI v1 tests work out of the box — their __main__ uses
        # nki.benchmark from neuronxcc.nki which is functional.
        is_v1 = _test_is_nki_v1(test_code)

        # Only patch __main__ for NKI v2 tests that need torch-based timing.
        if not is_v1 and "if __name__" in test_code_i:
            main_idx = test_code_i.index("if __name__")
            _MAIN_TEMPLATE = _NKI_V2_MAIN_TEMPLATE.replace(
                "__REF_FUNC__", ref_func_name
            )
            test_code_i = test_code_i[:main_idx] + _MAIN_TEMPLATE

        with open(temp_dir / f"code_{idx}.py", "w") as f:
            f.write(test_code_i)

        cmd = ["python", str(temp_dir.resolve() / f"code_{idx}.py")]
        env = None
        if core_id is not None:
            env = {**os.environ, "NEURON_RT_VISIBLE_CORES": str(core_id)}
        logger.info(f"Running command {' '.join(cmd)}")
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                               env=env)
        except subprocess.TimeoutExpired:
            logger.error(f"Code {idx} timed out after 600 seconds")
            return {
                "correct": False,
                "latency": None,
                "stdout": "",
                "stderr": "Timed out after 600 seconds",
            }

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

    def _try_combined_evaluation(
        self, test_code: str, code_strs: list[str], temp_dir: pathlib.Path
    ) -> list[dict] | None:
        """Two-phase evaluation: parallel compile then sequential eval.

        Phase 1 – Compile all implementations in parallel via nki.baremetal,
                  saving NEFFs to disk.
        Phase 2 – Load each NEFF via libnrt (instant, no recompilation),
                  run test_nki for correctness, and nrt_execute loop for
                  latency benchmarking.

        Returns None on infrastructure failure (caller falls back to
        per-implementation subprocess evaluation).
        """
        # Write implementation .py files
        self._write_impl_files(test_code, code_strs, temp_dir)

        # Copy nrt_helper.py into temp_dir so Phase 2 script can import it
        shutil.copy2(_NRT_HELPER_PATH, temp_dir / "nrt_helper.py")

        # Phase 1: parallel compilation (ref + all implementations)
        logger.info(
            f"Phase 1: compiling ref + {len(code_strs)} implementations in parallel"
        )
        compiled, compile_errors, neff_dir = self._compile_impls_parallel(
            test_code, code_strs, temp_dir
        )
        compiled_count = sum(1 for k, v in compiled.items() if v and k != "ref")
        logger.info(
            f"Phase 1 complete: ref={'OK' if compiled.get('ref') else 'FAILED'}, "
            f"{compiled_count}/{len(code_strs)} implementation NEFFs saved"
        )

        if not compiled.get("ref"):
            logger.error("Ref NEFF compilation failed, cannot run Phase 2")
            return None

        # Phase 2: sequential correctness + benchmark
        logger.info("Phase 2: correctness + benchmark via libnrt")
        phase2_script = self._generate_phase2_script(
            test_code, len(code_strs), temp_dir, neff_dir, compiled, compile_errors
        )
        phase2_path = temp_dir / "phase2_eval.py"
        with open(phase2_path, "w") as f:
            f.write(phase2_script)

        # ref compilation (~60s) + per-implementation libnrt eval (~10s each)
        timeout = 120 + 60 * len(code_strs)
        cmd = ["python", str(phase2_path.resolve())]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.error(f"Phase 2 timed out after {timeout}s")
            return None

        with open(temp_dir / "phase2_output.txt", "w") as f:
            f.write(f"=== STDOUT ===\n{p.stdout}\n=== STDERR ===\n{p.stderr}")

        results = self._parse_combined_results(p.stdout, len(code_strs))
        if results is None:
            logger.error(
                f"Phase 2 did not produce valid results. "
                f"returncode={p.returncode}, "
                f"stderr={p.stderr[:500]}"
            )
            return None

        for i, result in enumerate(results):
            if result["correct"]:
                logger.info(f"Code {i} latency: {result['latency']}")
            else:
                logger.error(f"Code {i} failed")

        return results

    def evaluate_code(
        self, prob: Prob, code_strs: list[str], simulator: str
    ) -> List[dict]:
        """Evaluate implementations using parallel NEFF compilation + libnrt.

        Phase 1 compiles all implementations in parallel (CPU-bound), saving
        NEFFs to disk.  Phase 2 loads each NEFF via libnrt (no
        recompilation) for correctness and latency benchmarking.

        Falls back to individual subprocess evaluation on infrastructure
        errors.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = pathlib.Path(__file__).parent / "tmp_files" / "trn_eval" / timestamp
        temp_dir.mkdir(parents=True, exist_ok=True)

        if prob.test_file:
            test_file = prob.test_file
        else:
            test_dir = TESTS_DIR / prob.prob_type
            matches = list(test_dir.glob(f"{prob.prob_id}_*.py"))
            if not matches:
                raise FileNotFoundError(
                    f"No test file found for {prob.prob_type} {prob.prob_id} "
                    f"in {test_dir}"
                )
            test_file = matches[0]
        test_code = test_file.read_text()

        results = None
        is_v1 = _test_is_nki_v1(test_code)

        # NKI v1: 2-phase (parallel baremetal compile + libnrt benchmark)
        if self.parallel and is_v1:
            results = self._try_combined_evaluation(test_code, code_strs, temp_dir)
        if results is not None:
            return results

        # NKI v2 (or v1 fallback): run _evaluate_single per implementation,
        # parallelized across NeuronCores.
        num_cores = self._detect_num_cores() if not is_v1 else 1
        max_parallel = min(num_cores, len(code_strs)) if num_cores > 0 else 1

        if max_parallel > 1:
            logger.info(f"Evaluating {len(code_strs)} implementations in parallel "
                        f"({max_parallel} cores)")
            results = [None] * len(code_strs)
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                futures = {}
                for i, code_str in enumerate(code_strs):
                    core_id = i % num_cores
                    futures[executor.submit(
                        self._evaluate_single, test_code, code_str,
                        temp_dir, i, core_id
                    )] = i
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
            return results

        results = []
        for i, code_str in enumerate(code_strs):
            results.append(self._evaluate_single(test_code, code_str, temp_dir, i))
        return results
