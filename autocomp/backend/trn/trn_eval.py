import os
import pathlib
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from datetime import datetime

from autocomp.common import logger, HARNESSES_DIR
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
        # NKI compiler expects base platform names (trn1, trn2, trn3, inf2)
        import re
        base_platform = re.match(r"(trn\d+|inf\d+)", platform)
        if base_platform:
            platform = base_platform.group(1)
        os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = platform
        logger.info(f"Auto-detected platform={platform} from instance={instance_type}")
    except Exception:
        logger.warning("Could not auto-detect NEURON_PLATFORM_TARGET_OVERRIDE from IMDS")


# Template for the __main__ block injected into NKI v2 test scripts.
# Runs correctness, compiles NEFF via CompileKernel (CPU-only), prints path.
# Actual benchmarking via neuron-profile happens in _evaluate_single after
# this subprocess exits and NeuronCores are freed.
# Placeholder __REF_FUNC__ is replaced at runtime.
_NKI_V2_MAIN_TEMPLATE = '''
if __name__ == "__main__":
    import os
    import time
    import tempfile
    import glob as _glob
    import numpy as np
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm

    _device = xm.xla_device()

    # Correctness test via torch on XLA
    test_result = test_nki(__REF_FUNC__, solution)
    if not test_result:
        print("Test failed")
        exit(1)
    else:
        print("Test passed")

    # Compile NEFF via CompileKernel (CPU-only, no NeuronCore needed).
    # Prints NEFF_PATH so the caller can benchmark via neuron-profile
    # after this process exits and NeuronCores are freed.
    _raw_fn = getattr(solution, "func", None)
    _bench_data_fn = _get_bench_data if "_get_bench_data" in dir() else None
    if _raw_fn is not None and _bench_data_fn is not None:
        try:
            from nki.framework.compiled import CompileKernel

            _ad = tempfile.mkdtemp(prefix="nki_bench_")
            _ck = CompileKernel(func=_raw_fn, artifacts_dir=_ad, _enable_simulation=True)

            _bd_cpu = _bench_data_fn(xm.xla_device())
            _NP_DTYPE_MAP = {
                'torch.bfloat16': np.float16, 'torch.float16': np.float16,
                'torch.float32': np.float32, 'torch.float64': np.float64,
                'torch.int32': np.int32, 'torch.int64': np.int64,
            }
            _np_inputs = []
            for _t in _bd_cpu:
                _ndt = _NP_DTYPE_MAP.get(str(_t.dtype), np.float32)
                _np_inputs.append(_t.detach().cpu().to(torch.float32).numpy().astype(_ndt))
            _ck_result = _ck(*_np_inputs)

            _neff = os.path.join(_ad, "kernel.neff")
            if os.path.exists(_neff):
                print(f"NEFF_PATH={_neff}")
        except Exception as _e:
            import traceback
            print(f"NEFF compilation failed: {_e}", flush=True)
            traceback.print_exc()

    # Fallback for non-NKI solutions (e.g. torch.einsum): use XLA wall-clock
    # timing with mark_step/wait_device_ops. torch_neuronx.trace splits the
    # graph into multiple NEFFs which profile unreliably; wall-clock captures
    # the end-to-end time the framework actually pays.
    if _raw_fn is None and _bench_data_fn is not None:
        try:
            _bd = _bench_data_fn(xm.xla_device())
            for _ in range(3):  # warmup
                _out = solution(*_bd)
                xm.mark_step(); xm.wait_device_ops()
            import time as _t
            _N = 20
            _t0 = _t.perf_counter()
            for _ in range(_N):
                _out = solution(*_bd)
                xm.mark_step()
            xm.wait_device_ops()
            _t1 = _t.perf_counter()
            _lat_ms = (_t1 - _t0) * 1000.0 / _N
            print(f"Latency: {_lat_ms:.3f} ms")
            print("NKI_BENCHMARK=true")
        except Exception as _e:
            import traceback
            print(f"wall-clock bench failed: {_e}", flush=True)
            traceback.print_exc()

    if _bench_data_fn is None:
        _cache_root = "/var/tmp/neuron-compile-cache"
        _all_neffs_post = _glob.glob(os.path.join(_cache_root, "**", "model.neff"), recursive=True)
        for _np in _all_neffs_post:
            print(f"NEFF_PATH={_np}")
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
        """Return the number of physical NeuronCores visible on this instance.

        neuron-ls nc_count reports logical NCs (e.g. 8 on Trn3 with LNC 2),
        but NEURON_RT_VISIBLE_CORES uses physical core IDs.
        Use neuroncore_ids list length for the physical count.
        """
        try:
            out = subprocess.check_output(
                ["neuron-ls", "--json-output"], text=True, timeout=5, stderr=subprocess.DEVNULL
            )
            data = json.loads(out)
            total_physical = sum(len(d.get("neuroncore_ids", [])) for d in data)
            if total_physical > 0:
                return total_physical
            return max(1, sum(d.get("nc_count", 0) for d in data))
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

    def _benchmark_via_neuron_profile(
        self, neff_path: str, temp_dir: pathlib.Path, idx,
        num_exec: int = 20, profile_nth: int = 10, core_id: int = None,
    ) -> float | None:
        """Benchmark a NEFF using neuron-profile capture + view.

        Runs the NEFF on-device num_exec times, profiles every profile_nth
        execution, and returns the steady-state total_time in milliseconds.
        """
        ntff_path = str(temp_dir / f"profile_{idx}.ntff")
        capture_cmd = [
            "neuron-profile", "capture",
            "-n", neff_path,
            "-s", ntff_path,
            "--num-exec", str(num_exec),
            "--profile-nth-exec", str(profile_nth),
        ]
        env = None
        if core_id is not None:
            env = {**os.environ, "NEURON_RT_VISIBLE_CORES": str(core_id)}
        try:
            p_cap = subprocess.run(
                capture_cmd, capture_output=True, text=True,
                timeout=120, env=env,
            )
        except subprocess.TimeoutExpired:
            logger.error(f"neuron-profile capture timed out for code {idx}")
            return None
        if p_cap.returncode != 0:
            logger.error(
                f"neuron-profile capture failed for code {idx}: "
                f"{p_cap.stderr[:300]}"
            )
            return None

        # neuron-profile writes <base>_exec_N.ntff (where base = session minus .ntff ext)
        ntff_base = ntff_path.rsplit(".ntff", 1)[0]
        steady_ntff = f"{ntff_base}_exec_{num_exec}.ntff"
        if not os.path.exists(steady_ntff):
            steady_ntff = f"{ntff_base}_exec_{profile_nth}.ntff"
        if not os.path.exists(steady_ntff):
            logger.error(f"No NTFF file found after profiling code {idx}")
            return None

        view_cmd = [
            "neuron-profile", "view",
            "-n", neff_path,
            "-s", steady_ntff,
            "--output-format=summary-json",
        ]
        try:
            p_view = subprocess.run(
                view_cmd, capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            logger.error(f"neuron-profile view timed out for code {idx}")
            return None

        for line in p_view.stdout.split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    key = list(data.keys())[0]
                    total_time_sec = data[key].get("total_time", 0)
                    total_time_ms = total_time_sec * 1000.0
                    logger.info(
                        f"Code {idx} neuron-profile total_time: "
                        f"{total_time_ms:.3f} ms"
                    )
                    return round(total_time_ms, 3)
                except (json.JSONDecodeError, IndexError, KeyError) as e:
                    logger.error(f"Failed to parse profile JSON: {e}")
                    return None
        logger.error(f"No JSON summary found in neuron-profile view output")
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

    def _ref_is_nki_kernel(self, test_code: str, ref_func_name: str) -> bool:
        """Return True if the ref function is decorated with @nki.jit.

        The BIRSim (v2) and baremetal (v1) parallel compile paths both
        require the ref to be an NKI kernel — they feed it through NKI's
        compiler to produce a NEFF. If ref is plain torch (e.g. einsum),
        Phase 1 will fail with parser errors on unsupported Python.
        """
        lines = test_code.split("\n")
        for i, line in enumerate(lines):
            if re.match(rf"\s*def\s+{re.escape(ref_func_name)}\s*\(", line):
                for j in range(max(0, i - 5), i):
                    if "@nki.jit" in lines[j] or "@baremetal" in lines[j]:
                        return True
                return False
        return False

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

        Uses neuronxcc.nki.baremetal(save_neff_name=...) to compile without
        needing a NeuronCore, enabling fully parallel CPU-only compilation.
        Only works for NKI v1 tests (neuronxcc.nki imports + numpy arrays).
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
_target_func = getattr(_mod, "solution", None)
if _target_func is None:
    print(json.dumps({{"compiled": False, "error": "No solution function defined"}}))
    sys.exit(0)
"""

        is_v2 = not _test_is_nki_v1(test_code)
        if is_v2:
            return self._compile_script_nki_v2(
                preamble, postamble, func_setup, neff_path, temp_dir,
            )
        return self._compile_script_baremetal(
            preamble, postamble, func_setup, neff_path, temp_dir,
        )

    def _compile_script_baremetal(
        self, preamble, postamble, func_setup, neff_path, temp_dir,
    ) -> str:
        """NKI v1 compile script: use neuronxcc.nki.baremetal to save NEFF."""
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

    def _compile_script_nki_v2(
        self, preamble, postamble, func_setup, neff_path, temp_dir,
    ) -> str:
        """NKI v2 (0.3.0) compile script: use CompileKernel(simulation=True).

        Compiles to NEFF via BIRSim on CPU, no NeuronCore required.
        Shims torch/XLA so test_nki() calls produce numpy arrays that
        CompileKernel can consume.
        """
        return f"""\
import sys, json, os, traceback
sys.path.insert(0, {repr(str(temp_dir.resolve()))})

# Load the torch/XLA shim from the helper module
sys.path.insert(0, {repr(str(pathlib.Path(__file__).parent.resolve()))})
from _v2_compile_shim import install_shim, OnceCompileKernel
install_shim()

import numpy as np

# === Test preamble ===
{preamble}
# === Test postamble ===
{postamble}

_neff_path = {repr(neff_path)}
{func_setup}

_raw = _target_func.func if hasattr(_target_func, "func") else _target_func
_wrapper = OnceCompileKernel(_raw, _neff_path)

_error_msg = ""
try:
    test_nki(_wrapper, _wrapper)
except Exception:
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
                if not np.array_equal(
                    _r.astype(np.float32), _c.astype(np.float32),
                ):
                    _passed = False
                    break
            if not _passed:
                break

        if not _passed:
            _result["stderr"] = "Test failed: output mismatch"
            _all_results[_idx] = _result
            continue

        _result["correct"] = True

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

    @staticmethod
    def _strip_v1_imports(code_str: str) -> str:
        """Remove neuronxcc.nki imports that would shadow NKI v2.

        Also strips duplicate nki/nl/nisa imports since the harness
        already provides them.
        """
        _DUP_IMPORT_PREFIXES = (
            "import neuronxcc.nki",
            "from neuronxcc.nki",
            "import neuronxcc",
            "import nki.language as",
            "import nki.isa as",
            "from nki.language import",
            "from nki.isa import",
            "from nki import",
        )
        lines = code_str.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(p) for p in _DUP_IMPORT_PREFIXES):
                continue
            if stripped == "import nki":
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    def _evaluate_single(
        self, test_code: str, code_str: str, temp_dir: pathlib.Path, idx: int,
        core_id: int = None,
    ) -> dict:
        """Evaluate a single implementation in its own subprocess.

        For NKI v2 tests, runs correctness in a subprocess then uses
        neuron-profile to benchmark the compiled NEFF separately.
        For NKI v1 tests, runs as-is.
        """
        is_v1 = _test_is_nki_v1(test_code)
        if not is_v1:
            code_str = self._strip_v1_imports(code_str)
        test_code_i = test_code.replace("# SUBSTITUTE HERE", code_str)
        ref_func_name = self._extract_ref_func_name(test_code)

        is_v1 = _test_is_nki_v1(test_code)

        # Only patch __main__ for NKI v2 tests.
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

        result_dict = {
            "correct": False,
            "latency": None,
            "stdout": p.stdout,
            "stderr": p.stderr,
        }

        if p.returncode != 0:
            with open(temp_dir / f"code_{idx}_output.txt", "w") as f:
                f.write("=== STDOUT ===\n" + p.stdout + "\n=== STDERR ===\n" + p.stderr)
            logger.error(f"Code {idx} failed to run")
            return result_dict

        if "Test passed" not in p.stdout:
            with open(temp_dir / f"code_{idx}_output.txt", "w") as f:
                f.write("=== STDOUT ===\n" + p.stdout + "\n=== STDERR ===\n" + p.stderr)
            logger.error(f"Code {idx} correctness test failed")
            return result_dict

        # For NKI v2: benchmark via neuron-profile now that the subprocess
        # has exited and NeuronCores are free.
        # Skip if the template already benchmarked via SpikeModel.
        if not is_v1 and "NKI_BENCHMARK=true" not in p.stdout:
            # Discover NEFFs for this candidate.
            # Primary: mtime-based NEFF_PATH from template (most reliable
            # since the template primes the ref kernel before snapshotting).
            combined_output = p.stdout + "\n" + p.stderr
            neff_paths = []

            for line in p.stdout.split("\n"):
                if line.startswith("NEFF_PATH="):
                    path = line.split("=", 1)[1].strip()
                    if path and path not in neff_paths:
                        neff_paths.append(path)

            # Fallback: parse Neuron runtime log for compiled NEFF paths
            # (useful when the mtime approach misses NEFFs, e.g. NKI v1).
            if not neff_paths:
                cached = re.findall(
                    r'Using a cached neff at\s+(\S+model\.neff)', combined_output
                )
                compiled_mods = re.findall(
                    r'Compilation Successfully Completed for model\.(MODULE_\d+\+\w+)\.',
                    combined_output,
                )
                compiled_paths = []
                for mod in dict.fromkeys(compiled_mods):
                    import glob as _g
                    cands = _g.glob(os.path.join(
                        "/var/tmp/neuron-compile-cache", "**",
                        mod, "model.neff",
                    ), recursive=True)
                    if cands:
                        compiled_paths.append(cands[0])

                for p_ in cached + compiled_paths:
                    if p_ not in neff_paths:
                        neff_paths.append(p_)

            # Profile all discovered NEFFs and sum their total_times.
            # A single candidate can produce multiple NEFFs (graph splits via
            # mark_step, etc.) that run sequentially — the total on-device
            # cost is the sum.
            neff_paths = [p_ for p_ in neff_paths if os.path.exists(p_)]
            if neff_paths:
                total_latency = 0.0
                all_profiled = True
                for ni, neff_p in enumerate(neff_paths):
                    lat = self._benchmark_via_neuron_profile(
                        neff_p, temp_dir, f"{idx}_neff{ni}", core_id=core_id
                    )
                    if lat is not None:
                        total_latency += lat
                    else:
                        all_profiled = False
                if all_profiled and total_latency > 0:
                    total_latency = round(total_latency, 3)
                    p_stdout_with_latency = p.stdout + f"\nLatency: {total_latency:.3f} ms\n"
                    result_dict["stdout"] = p_stdout_with_latency
                if len(neff_paths) > 1:
                    logger.warning(
                        f"Code {idx}: candidate produced {len(neff_paths)} NEFFs "
                        f"(graph split). Summed latency={total_latency}ms. "
                        f"Single-NEFF candidates are preferred."
                    )
            else:
                logger.warning(f"Code {idx}: could not find NEFF for profiling")

        with open(temp_dir / f"code_{idx}_output.txt", "w") as f:
            f.write("=== STDOUT ===\n" + result_dict["stdout"] + "\n=== STDERR ===\n" + p.stderr)

        result_dict["correct"] = True
        latency = self._extract_latency(result_dict["stdout"])
        if latency is None:
            logger.error(f"Code {idx} did not produce latency output")
            return result_dict

        logger.info(f"Code {idx} latency: {latency}")
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

        # Phase 1 (BIRSim CPU compile / baremetal) requires ref to be an NKI
        # kernel. If ref is plain torch (e.g. torch.einsum for the LM head
        # baseline), skip Phase 1 and fall back to per-core _evaluate_single.
        ref_func_name = self._extract_ref_func_name(test_code)
        if not self._ref_is_nki_kernel(test_code, ref_func_name):
            logger.info(
                f"Ref '{ref_func_name}' is not an NKI kernel (no @nki.jit); "
                "skipping parallel compile and using per-core evaluation"
            )
            return None

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

        # Phase 2: correctness via libnrt (no NKI recompilation)
        logger.info("Phase 2: correctness via libnrt")
        phase2_script = self._generate_phase2_script(
            test_code, len(code_strs), temp_dir, neff_dir, compiled, compile_errors
        )
        phase2_path = temp_dir / "phase2_eval.py"
        with open(phase2_path, "w") as f:
            f.write(phase2_script)

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

        # Phase 3: benchmark passing candidates via neuron-profile (accurate
        # on-device latency from hardware counters, same as _evaluate_single).
        # Run profiles in parallel across NeuronCores.
        passing = [
            (i, str((neff_dir / f"impl_{i}.neff").resolve()))
            for i, r in enumerate(results)
            if r["correct"] and os.path.exists(
                str((neff_dir / f"impl_{i}.neff").resolve()))
        ]
        for i, r in enumerate(results):
            if not r["correct"]:
                logger.error(f"Code {i} failed correctness")

        if passing:
            # Phase 3 runs sequentially: LNC=2 kernels can occupy both logical
            # NCs and neuron-profile's core-pinning via NEURON_RT_VISIBLE_CORES
            # has been racy in practice (lnc allocation conflicts between
            # concurrently-running profiles). The speedup from parallel
            # profiling is small anyway — each call is O(hundreds of ms).
            def _profile_one(idx, neff_path, core_id):
                lat = self._benchmark_via_neuron_profile(
                    neff_path, temp_dir, f"{idx}_combined", core_id=core_id
                )
                if lat is not None:
                    results[idx]["latency"] = round(lat, 3)
                    results[idx]["stdout"] = f"Latency: {lat:.3f} ms\n"
                    logger.info(f"Code {idx} latency: {results[idx]['latency']}")
                else:
                    logger.warning(f"Code {idx}: neuron-profile failed")

            for idx, neff_path in passing:
                _profile_one(idx, neff_path, core_id=0)

        return results

    def evaluate_code(
        self, prob: Prob, code_strs: list[str], simulator: str
    ) -> List[dict]:
        """Evaluate implementations, parallelizing across NeuronCores.

        Each candidate runs correctness + neuron-profile profiling on a
        dedicated NeuronCore.  Candidates assigned to the same core run
        sequentially (profiling needs exclusive core access), but
        different cores operate in parallel.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = pathlib.Path(__file__).parent / "tmp_files" / "trn_eval" / timestamp
        temp_dir.mkdir(parents=True, exist_ok=True)

        if prob.test_file:
            test_file = prob.test_file
        else:
            test_dir = HARNESSES_DIR / prob.prob_type
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

        # 2-phase: parallel compile (baremetal for v1, BIRSim for v2) + libnrt benchmark
        if self.parallel:
            results = self._try_combined_evaluation(test_code, code_strs, temp_dir)
        if results is not None:
            return results

        # NKI v2: parallelize across NeuronCores.
        num_cores = self._detect_num_cores()
        logger.info(
            f"Evaluating {len(code_strs)} candidates across {num_cores} NeuronCore(s)"
        )

        results = [None] * len(code_strs)

        # Partition candidates into per-core queues (round-robin)
        core_queues: list[list[int]] = [[] for _ in range(num_cores)]
        for i in range(len(code_strs)):
            core_queues[i % num_cores].append(i)

        def _run_core_queue(core_id: int, indices: list[int]):
            """Run all candidates assigned to this core sequentially."""
            for idx in indices:
                results[idx] = self._evaluate_single(
                    test_code, code_strs[idx], temp_dir, idx, core_id=core_id
                )

        if num_cores <= 1 or len(code_strs) <= 1:
            _run_core_queue(0, list(range(len(code_strs))))
        else:
            with ThreadPoolExecutor(max_workers=num_cores) as executor:
                futures = []
                for core_id, indices in enumerate(core_queues):
                    if indices:
                        futures.append(
                            executor.submit(_run_core_queue, core_id, indices)
                        )
                for f in futures:
                    f.result()

        return results
