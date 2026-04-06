import numpy as np
import math

import nki
import nki.language as nl
import nki.isa as nisa
import torch
from torch_xla.core import xla_model as xm


# Hardware tile limits for NeuronCore-v3
TILE_K = 128
TILE_M = 128
TILE_N = 512


# SUBSTITUTE HERE


@nki.jit
def ref(a_t_hbm, b_hbm):
    """Reference: identical to test (tiled GEMM)."""
    K = a_t_hbm.shape[0]
    M = a_t_hbm.shape[1]
    N = b_hbm.shape[1]

    c_hbm = nl.ndarray((M, N), dtype=a_t_hbm.dtype, buffer=nl.hbm)

    num_k_tiles = K // TILE_K
    num_m_tiles = (M + TILE_M - 1) // TILE_M
    num_n_tiles = (N + TILE_N - 1) // TILE_N

    for m_tile_idx in nl.affine_range(num_m_tiles):
        m_start = m_tile_idx * TILE_M
        m_size = min(M - m_start, TILE_M)

        for n_tile_idx in nl.affine_range(num_n_tiles):
            n_start = n_tile_idx * TILE_N
            n_size = min(N - n_start, TILE_N)

            result_psum = nl.ndarray((m_size, n_size), dtype=nl.float32, buffer=nl.psum)

            for k_idx in nl.sequential_range(num_k_tiles):
                a_tile = nl.ndarray(
                    (TILE_K, m_size), dtype=a_t_hbm.dtype, buffer=nl.sbuf
                )
                nisa.dma_copy(
                    dst=a_tile,
                    src=a_t_hbm[
                        k_idx * TILE_K : (k_idx + 1) * TILE_K,
                        m_start : m_start + m_size,
                    ],
                )

                b_tile = nl.ndarray((TILE_K, n_size), dtype=b_hbm.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=b_tile,
                    src=b_hbm[
                        k_idx * TILE_K : (k_idx + 1) * TILE_K,
                        n_start : n_start + n_size,
                    ],
                )

                nisa.nc_matmul(
                    dst=result_psum,
                    stationary=a_tile,
                    moving=b_tile,
                )

            result_sbuf = nl.ndarray(
                (m_size, n_size), dtype=a_t_hbm.dtype, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=result_sbuf, src=result_psum)
            nisa.dma_copy(
                dst=c_hbm[m_start : m_start + m_size, n_start : n_start + n_size],
                src=result_sbuf,
            )

    return c_hbm


def test_nki(ref_func, test_func):
    """Correctness check: compare ref and test vs CPU matmul."""
    device = xm.xla_device()

    # Test shape: M=128, K=1024, N=1024 (all tile-aligned)
    M, K, N = 128, 1024, 1024

    for seed in range(2):
        np.random.seed(42 + seed)
        a_np = (np.random.randn(M, K) * 0.1).astype(np.float32)
        b_np = (np.random.randn(K, N) * 0.1).astype(np.float32)

        # CPU reference
        c_cpu = a_np @ b_np

        # Prepare A^T for kernel (transposed on host)
        a_t_np = a_np.T.copy()  # [K, M]

        a_t_dev = torch.tensor(a_t_np, dtype=torch.bfloat16, device=device)
        b_dev = torch.tensor(b_np, dtype=torch.bfloat16, device=device)

        result_ref = ref_func(a_t_dev, b_dev)
        result_test = test_func(a_t_dev, b_dev)

        ref_out = result_ref.detach().cpu().float().numpy()
        test_out = result_test.detach().cpu().float().numpy()

        # NKI ref vs NKI test
        cos_ref_test = np.dot(ref_out.flatten(), test_out.flatten()) / (
            np.linalg.norm(ref_out.flatten()) * np.linalg.norm(test_out.flatten())
            + 1e-12
        )

        # NKI test vs CPU
        cos_test_cpu = np.dot(test_out.flatten(), c_cpu.flatten()) / (
            np.linalg.norm(test_out.flatten()) * np.linalg.norm(c_cpu.flatten()) + 1e-12
        )

        print(
            f"  seed={42 + seed}: cos(ref,test)={cos_ref_test:.6f}, cos(test,cpu)={cos_test_cpu:.6f}"
        )

        if cos_ref_test < 0.99:
            print(f"  FAIL: NKI ref vs test cosine {cos_ref_test:.6f} < 0.99")
            return False

        # bfloat16 matmul has lower precision, so use 0.99 threshold
        if cos_test_cpu < 0.99:
            print(f"  FAIL: NKI test vs CPU cosine {cos_test_cpu:.6f} < 0.99")
            return False

    return True


def benchmark_nki(nki_func):
    """Latency benchmark using nki.benchmark (monkey-patched by trn_eval.py)."""
    device = xm.xla_device()
    M, K, N = 128, 1024, 1024

    np.random.seed(42)
    a_t_np = (np.random.randn(K, M) * 0.1).astype(np.float32)
    b_np = (np.random.randn(K, N) * 0.1).astype(np.float32)

    a_t_dev = torch.tensor(a_t_np, dtype=torch.bfloat16, device=device)
    b_dev = torch.tensor(b_np, dtype=torch.bfloat16, device=device)

    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(a_t_dev, b_dev)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))


if __name__ == "__main__":
    test_result = test_nki(ref, solution)
    if not test_result:
        print("Test failed")
        exit(1)
    else:
        print("Test passed")
        benchmark_nki(solution)
