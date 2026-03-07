import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


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


def _print_hw_feedback(*, M: int, K: int, N: int, dtype, latency_ms: float) -> None:
    flops = 2.0 * M * K * N
    secs = max(1e-9, latency_ms / 1000.0)
    achieved_tflops = flops / secs / 1e12
    peak_tflops = max(1e-9, _default_peak_tflops(dtype))
    util_pct = max(0.0, min(100.0, achieved_tflops / peak_tflops * 100.0))

    print(f"Utilization: {util_pct:.2f}%")
    print(f"Achieved TFLOP/s: {achieved_tflops:.3f} (peak={peak_tflops:.3f})")


def _run_autocomp_harness():
    M = 1024
    K = 1024
    N = 1024
    num_trials = int(os.getenv("AUTOCOMP_TPU_NUM_TRIALS", "10"))
    num_warmup = int(os.getenv("AUTOCOMP_TPU_NUM_WARMUP", "3"))
    dtype_str = os.getenv("AUTOCOMP_TPU_DTYPE", "float32")
    dtype = getattr(jnp, dtype_str, jnp.float32)

    x = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=dtype)
    y = jax.random.normal(jax.random.PRNGKey(1), (K, N), dtype=dtype)

    print(f"Running TPU test() ({M}x{K}x{N}, dtype={dtype_str}, "
          f"warmup={num_warmup}, trials={num_trials})...")

    # Warmup: compile + stabilize.
    for _ in range(num_warmup):
        out = test(x, y)
        jax.block_until_ready(out)

    # Correctness check.
    baseline = jnp.matmul(x, y)
    if not jnp.allclose(baseline, out, atol=1e-3, rtol=1e-3):
        print("FAIL: Verification failed.")
        raise SystemExit(1)

    # Timed runs.
    times_ms = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        out2 = test(x, y)
        jax.block_until_ready(out2)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms.sort()
    latency_ms = times_ms[len(times_ms) // 2]  # median
    print(f"Latency: {latency_ms:.3f} ms")
    print(f"  all trials (ms): {[f'{t:.3f}' for t in times_ms]}")
    _print_hw_feedback(M=M, K=K, N=N, dtype=dtype, latency_ms=latency_ms)
    print("SUCCESS: test() matches JAX baseline.")


if __name__ == "__main__":
    _run_autocomp_harness()
