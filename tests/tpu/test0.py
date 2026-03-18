"""TPU matmul test harness for Autocomp.

The batch evaluator exec's each implementation and expects it to define a
`test()` function, then calls `_run_autocomp_harness()` to benchmark it.

To add a new TPU problem, copy this file as test{prob_id}.py and adapt:
  - Problem dimensions and dtype
  - _create_inputs() to generate the right input tensors
  - _reference() with the baseline computation
  - _compute_util() if you want utilization feedback
"""
import os
import time
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# ---------------------------------------------------------------------------
# Problem configuration
# ---------------------------------------------------------------------------
M = 1024
K = 1024
N = 1024
DTYPE_STR = os.getenv("AUTOCOMP_TPU_DTYPE", "float32")
DTYPE = getattr(jnp, DTYPE_STR, jnp.float32)

NUM_WARMUP = int(os.getenv("AUTOCOMP_TPU_NUM_WARMUP", "3"))
NUM_TRIALS = int(os.getenv("AUTOCOMP_TPU_NUM_TRIALS", "10"))

# ---------------------------------------------------------------------------
# Inputs and reference — adapt these for each new problem
# ---------------------------------------------------------------------------

def _create_inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=DTYPE)
    y = jax.random.normal(jax.random.PRNGKey(1), (K, N), dtype=DTYPE)
    return (x, y)


def _reference(x, y):
    return jnp.matmul(x, y)


def _default_peak_tflops(dtype) -> float:
    override = os.getenv("AUTOCOMP_TPU_PEAK_TFLOPS")
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    dt = jnp.dtype(dtype)
    if dt in (jnp.bfloat16, jnp.float16):
        return float(os.getenv("AUTOCOMP_TPU_V6E_BF16_TFLOPS", "918"))
    if dt == jnp.float32:
        return float(os.getenv("AUTOCOMP_TPU_V6E_FP32_TFLOPS", "459"))
    return float(os.getenv("AUTOCOMP_TPU_V6E_BF16_TFLOPS", "918"))


def _compute_util(latency_ms: float) -> None:
    """Print utilization feedback. Remove or adapt per problem."""
    flops = 2.0 * M * K * N
    secs = max(1e-9, latency_ms / 1000.0)
    achieved_tflops = flops / secs / 1e12
    peak_tflops = max(1e-9, _default_peak_tflops(DTYPE))
    util_pct = max(0.0, min(100.0, achieved_tflops / peak_tflops * 100.0))
    print(f"Utilization: {util_pct:.2f}%")
    print(f"Achieved TFLOP/s: {achieved_tflops:.3f} (peak={peak_tflops:.3f})")

# ---------------------------------------------------------------------------
# Harness — common logic, usually no need to modify
# ---------------------------------------------------------------------------

def _run_autocomp_harness():
    inputs = _create_inputs()

    print(f"Running TPU test() ({M}x{K}x{N} matmul, dtype={DTYPE_STR}, "
          f"warmup={NUM_WARMUP}, trials={NUM_TRIALS})...")

    # Warmup
    for _ in range(NUM_WARMUP):
        out = test(*inputs)
        jax.block_until_ready(out)

    # Correctness
    ref_out = _reference(*inputs)
    if not jnp.allclose(ref_out, out, atol=1e-3, rtol=1e-3):
        max_diff = float(jnp.max(jnp.abs(ref_out - out)))
        print(f"FAIL: Verification failed (max_diff={max_diff:.6f}).")
        raise SystemExit(1)

    # Benchmark
    times_ms = []
    for _ in range(NUM_TRIALS):
        t0 = time.perf_counter()
        out = test(*inputs)
        jax.block_until_ready(out)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms.sort()
    latency_ms = times_ms[len(times_ms) // 2]
    print(f"Latency: {latency_ms:.3f} ms")
    print(f"  all trials (ms): {[f'{t:.3f}' for t in times_ms]}")
    _compute_util(latency_ms)
    print("SUCCESS: test() matches reference.")


if __name__ == "__main__":
    _run_autocomp_harness()
