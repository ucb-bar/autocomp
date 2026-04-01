import numpy as np
import math

import nki
import nki.language as nl
import nki.isa as nisa
import torch
from torch_xla.core import xla_model as xm


TILE_P = 128
TILE_S = 128
TILE_M = 512


# SUBSTITUTE HERE


@nki.jit
def test(a_ki, b_kj):
    """Single-channel GEMM: result[i,j] = sum_k a[k,i]^T * b[k,j]

    Args:
        a_ki: [N, N] -- lhsT layout (K=dim0, I=dim1)
        b_kj: [N, N] -- rhs layout (K=dim0, J=dim1)
    Returns:
        result: [N, N] -- (I=dim0, J=dim1)
    """
    N = a_ki.shape[0]

    n_k = N // TILE_P
    n_i = N // TILE_S
    n_j = N // TILE_M if N >= TILE_M else 1
    jtile = TILE_M if N >= TILE_M else N

    result = nl.ndarray((N, N), dtype=a_ki.dtype, buffer=nl.shared_hbm)

    for i_t in nl.affine_range(n_i):
        i0 = i_t * TILE_S

        for j_t in nl.affine_range(n_j):
            j0 = j_t * jtile

            acc = nl.ndarray((TILE_S, jtile), dtype=nl.float32, buffer=nl.psum)

            for k_t in nl.sequential_range(n_k):
                k0 = k_t * TILE_P

                lhsT_tile = nl.ndarray(
                    (TILE_P, TILE_S), dtype=a_ki.dtype, buffer=nl.sbuf
                )
                nisa.dma_copy(
                    dst=lhsT_tile, src=a_ki[k0 : k0 + TILE_P, i0 : i0 + TILE_S]
                )

                rhs_tile = nl.ndarray((TILE_P, jtile), dtype=b_kj.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=rhs_tile, src=b_kj[k0 : k0 + TILE_P, j0 : j0 + jtile])

                nisa.nc_matmul(dst=acc, stationary=lhsT_tile, moving=rhs_tile)

            out = nl.ndarray((TILE_S, jtile), dtype=a_ki.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out, src=acc)
            nisa.dma_copy(dst=result[i0 : i0 + TILE_S, j0 : j0 + jtile], src=out)

    return result


@nki.jit
def ref(a_ki, b_kj):
    """Reference: identical to test (single-channel GEMM)."""
    N = a_ki.shape[0]

    n_k = N // TILE_P
    n_i = N // TILE_S
    n_j = N // TILE_M if N >= TILE_M else 1
    jtile = TILE_M if N >= TILE_M else N

    result = nl.ndarray((N, N), dtype=a_ki.dtype, buffer=nl.shared_hbm)

    for i_t in nl.affine_range(n_i):
        i0 = i_t * TILE_S

        for j_t in nl.affine_range(n_j):
            j0 = j_t * jtile

            acc = nl.ndarray((TILE_S, jtile), dtype=nl.float32, buffer=nl.psum)

            for k_t in nl.sequential_range(n_k):
                k0 = k_t * TILE_P

                lhsT_tile = nl.ndarray(
                    (TILE_P, TILE_S), dtype=a_ki.dtype, buffer=nl.sbuf
                )
                nisa.dma_copy(
                    dst=lhsT_tile, src=a_ki[k0 : k0 + TILE_P, i0 : i0 + TILE_S]
                )

                rhs_tile = nl.ndarray((TILE_P, jtile), dtype=b_kj.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=rhs_tile, src=b_kj[k0 : k0 + TILE_P, j0 : j0 + jtile])

                nisa.nc_matmul(dst=acc, stationary=lhsT_tile, moving=rhs_tile)

            out = nl.ndarray((TILE_S, jtile), dtype=a_ki.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out, src=acc)
            nisa.dma_copy(dst=result[i0 : i0 + TILE_S, j0 : j0 + jtile], src=out)

    return result


def test_nki(ref_func, test_func):
    """Correctness check: compare ref and test vs CPU matmul."""
    device = xm.xla_device()

    # N=512: 4 tiles along each I/J dimension, good coverage of tiling logic
    N = 512

    for seed in range(2):
        np.random.seed(42 + seed)
        # a_ki is in lhsT layout [K, I] = [N, N]
        a_np = (np.random.randn(N, N) * 0.1).astype(np.float32)
        b_np = (np.random.randn(N, N) * 0.1).astype(np.float32)

        # CPU reference: result = a^T @ b
        c_cpu = a_np.T @ b_np

        a_dev = torch.tensor(a_np, dtype=torch.bfloat16, device=device)
        b_dev = torch.tensor(b_np, dtype=torch.bfloat16, device=device)

        result_ref = ref_func(a_dev, b_dev)
        result_test = test_func(a_dev, b_dev)

        ref_out = result_ref.detach().cpu().float().numpy()
        test_out = result_test.detach().cpu().float().numpy()

        cos_ref_test = np.dot(ref_out.flatten(), test_out.flatten()) / (
            np.linalg.norm(ref_out.flatten()) * np.linalg.norm(test_out.flatten())
            + 1e-12
        )

        cos_test_cpu = np.dot(test_out.flatten(), c_cpu.flatten()) / (
            np.linalg.norm(test_out.flatten()) * np.linalg.norm(c_cpu.flatten()) + 1e-12
        )

        print(
            f"  seed={42 + seed}: cos(ref,test)={cos_ref_test:.6f}, cos(test,cpu)={cos_test_cpu:.6f}"
        )

        if cos_ref_test < 0.99:
            print(f"  FAIL: NKI ref vs test cosine {cos_ref_test:.6f} < 0.99")
            return False

        if cos_test_cpu < 0.99:
            print(f"  FAIL: NKI test vs CPU cosine {cos_test_cpu:.6f} < 0.99")
            return False

    return True


def benchmark_nki(nki_func):
    """Latency benchmark using nki.benchmark (monkey-patched by trn_eval.py)."""
    device = xm.xla_device()
    N = 512

    np.random.seed(42)
    a_np = (np.random.randn(N, N) * 0.1).astype(np.float32)
    b_np = (np.random.randn(N, N) * 0.1).astype(np.float32)

    a_dev = torch.tensor(a_np, dtype=torch.bfloat16, device=device)
    b_dev = torch.tensor(b_np, dtype=torch.bfloat16, device=device)

    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(a_dev, b_dev)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))


if __name__ == "__main__":
    test_result = test_nki(ref, test)
    if not test_result:
        print("Test failed")
        exit(1)
    else:
        print("Test passed")
        benchmark_nki(test)
