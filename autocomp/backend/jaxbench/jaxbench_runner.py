"""
JAXBench runner — executed on the TPU VM by autocomp.

Usage:
    python jaxbench_runner.py <workload.py> <impl_0.py> [impl_1.py ...]

For each implementation file, imports its `workload()` function, checks correctness
against the reference workload from the workload file, benchmarks, and prints
delimited JSON results to stdout.
"""
import importlib.util
import json
import multiprocessing
import os
import sys
import time
import traceback

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl  # noqa: F401 — available for implementations
from jax.experimental.pallas import tpu as pltpu  # noqa: F401

DELIM_START = "===JAXBENCH_IMPL_START==="
DELIM_END = "===JAXBENCH_IMPL_END==="

NUM_WARMUP = int(os.getenv("AUTOCOMP_TPU_NUM_WARMUP", "5"))
NUM_TRIALS = int(os.getenv("AUTOCOMP_TPU_NUM_TRIALS", "100"))
ATOL = float(os.getenv("AUTOCOMP_JAXBENCH_ATOL", "3.125e-2"))
RTOL = float(os.getenv("AUTOCOMP_JAXBENCH_RTOL", "1e-2"))
IMPL_TIMEOUT = int(os.getenv("AUTOCOMP_JAXBENCH_IMPL_TIMEOUT", "120"))


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eval_impl(impl_path: str, inputs, ref_out):
    """Evaluate a single implementation. Returns a result dict."""
    result = {"correct": False, "latency": None, "error": ""}

    try:
        impl_mod = _load_module(impl_path, "impl")
    except Exception:
        result["error"] = traceback.format_exc()
        return result

    impl_workload = getattr(impl_mod, "workload", None)
    if impl_workload is None:
        result["error"] = "implementation does not define a workload() function"
        return result

    try:
        impl_fn = jax.jit(impl_workload)
        for _ in range(NUM_WARMUP):
            impl_out = impl_fn(*inputs)
            jax.block_until_ready(impl_out)

        if not jnp.allclose(ref_out, impl_out, atol=ATOL, rtol=RTOL):
            max_diff = float(jnp.max(jnp.abs(ref_out - impl_out)))
            result["error"] = f"correctness check failed (max_diff={max_diff:.6f})"
            return result

        max_diff = float(jnp.max(jnp.abs(ref_out - impl_out)))
        denom = jnp.maximum(jnp.max(jnp.abs(ref_out)), 1e-6)
        max_rel_diff = float(jnp.max(jnp.abs(ref_out - impl_out) / denom))

        times_ms = []
        for _ in range(NUM_TRIALS):
            t0 = time.perf_counter()
            out = impl_fn(*inputs)
            jax.block_until_ready(out)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        times_ms.sort()
        result["correct"] = True
        result["latency"] = round(times_ms[len(times_ms) // 2], 3)
        result["all_times_ms"] = [round(t, 3) for t in times_ms]
        result["max_diff"] = round(max_diff, 6)
        result["max_rel_diff"] = round(max_rel_diff, 6)

    except SystemExit:
        result["error"] = "SystemExit raised"
    except Exception:
        result["error"] = traceback.format_exc()

    return result


def _eval_worker(impl_path, workload_path, result_queue):
    """Run in a subprocess so we can hard-kill on timeout."""
    try:
        ref_mod = _load_module(workload_path, "jaxbench_ref")
        if hasattr(ref_mod, "create_inputs") and hasattr(ref_mod, "workload"):
            inputs = ref_mod.create_inputs()
            ref_fn = jax.jit(ref_mod.workload)
        elif hasattr(ref_mod, "Model") and hasattr(ref_mod, "get_inputs"):
            init_args = ref_mod.get_init_inputs() if hasattr(ref_mod, "get_init_inputs") else []
            model = ref_mod.Model(*init_args)
            inputs = ref_mod.get_inputs()
            ref_fn = jax.jit(model.forward)
        else:
            result_queue.put({"correct": False, "latency": None, "error": "bad workload"})
            return
        ref_out = ref_fn(*inputs)
        jax.block_until_ready(ref_out)
        result = _eval_impl(impl_path, inputs, ref_out)
    except Exception:
        result = {"correct": False, "latency": None, "error": traceback.format_exc()}
    result_queue.put(result)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <workload.py> <impl_0.py> [impl_1.py ...]",
              file=sys.stderr)
        sys.exit(1)

    workload_path = sys.argv[1]
    impl_paths = sys.argv[2:]

    for idx, impl_path in enumerate(impl_paths):
        print(DELIM_START, flush=True)

        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()
        proc = ctx.Process(target=_eval_worker,
                           args=(impl_path, workload_path, result_queue))
        proc.start()
        proc.join(timeout=IMPL_TIMEOUT)

        if proc.is_alive():
            proc.kill()
            proc.join(timeout=10)
            result = {"correct": False, "latency": None,
                      "error": f"timed out after {IMPL_TIMEOUT}s"}
        elif not result_queue.empty():
            result = result_queue.get_nowait()
        else:
            result = {"correct": False, "latency": None,
                      "error": f"worker exited with code {proc.exitcode}"}

        if result["correct"]:
            print(f"Latency: {result['latency']:.3f} ms")
        elif result["error"]:
            print(f"FAIL: {result['error']}")

        print(json.dumps(result))
        print(DELIM_END, flush=True)


if __name__ == "__main__":
    main()
