import numpy as np
import math

import nki
import nki.language as nl
import nki.isa as nisa
import torch
from torch_xla.core import xla_model as xm


# SUBSTITUTE HERE


@nki.jit
def ref(
    a,
    b,
):
    """Reference: identical to test (triangular multiply, Boltz-2 pairformer)."""
    N, N2, D = a.shape
    N_b, N2_b, D_b = b.shape
    P_MAX = 128

    assert N == N2
    assert N == N_b and N2 == N2_b and D == D_b
    assert N % P_MAX == 0
    assert D <= P_MAX

    output = nl.ndarray((N, N, D), dtype=a.dtype, buffer=nl.shared_hbm)

    stride_i = N * D
    stride_k = D

    n_tiles = N // P_MAX

    for d in nl.affine_range(D):
        for i_tile in nl.affine_range(n_tiles):
            i_start = i_tile * P_MAX

            for j_tile in nl.affine_range(n_tiles):
                j_start = j_tile * P_MAX

                acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=acc, value=0.0)

                for k_tile_idx in nl.sequential_range(n_tiles):
                    k_start = k_tile_idx * P_MAX

                    a_tile = nl.ndarray((P_MAX, P_MAX), dtype=a.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=a_tile,
                        src=a.ap(
                            pattern=[[stride_i, P_MAX], [stride_k, P_MAX]],
                            offset=i_start * stride_i + k_start * stride_k + d,
                        ),
                    )

                    b_tile = nl.ndarray((P_MAX, P_MAX), dtype=b.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=b_tile,
                        src=b.ap(
                            pattern=[[stride_i, P_MAX], [stride_k, P_MAX]],
                            offset=j_start * stride_i + k_start * stride_k + d,
                        ),
                    )

                    a_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=a.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=a_t_psum, data=a_tile)
                    a_t = nl.ndarray((P_MAX, P_MAX), dtype=a.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=a_t, src=a_t_psum)

                    b_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=b.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=b_t_psum, data=b_tile)
                    b_t = nl.ndarray((P_MAX, P_MAX), dtype=b.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=b_t, src=b_t_psum)

                    partial_psum = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum
                    )
                    nisa.nc_matmul(dst=partial_psum, stationary=a_t, moving=b_t)
                    partial = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=partial, src=partial_psum)

                    nisa.tensor_tensor(dst=acc, data1=acc, data2=partial, op=nl.add)

                out_tile = nl.ndarray((P_MAX, P_MAX), dtype=a.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=out_tile, src=acc)

                nisa.dma_copy(
                    dst=output.ap(
                        pattern=[[stride_i, P_MAX], [stride_k, P_MAX]],
                        offset=i_start * stride_i + j_start * stride_k + d,
                    ),
                    src=out_tile,
                )

    return output


def test_nki(ref_func, test_func):
    """Correctness check: compare ref and test on device tensors."""
    device = xm.xla_device()
    N = 128
    D = 128

    for seed in range(2):
        np.random.seed(42 + seed)
        a_np = (np.random.randn(N, N, D) * 0.1).astype(np.float32)
        b_np = (np.random.randn(N, N, D) * 0.1).astype(np.float32)

        a_t = torch.tensor(a_np, dtype=torch.bfloat16, device=device)
        b_t = torch.tensor(b_np, dtype=torch.bfloat16, device=device)

        result_ref = ref_func(a_t, b_t)
        result_test = test_func(a_t, b_t)

        ref_out = result_ref.detach().cpu().float().numpy()
        test_out = result_test.detach().cpu().float().numpy()

        cos_sim = np.dot(ref_out.flatten(), test_out.flatten()) / (
            np.linalg.norm(ref_out.flatten()) * np.linalg.norm(test_out.flatten())
            + 1e-12
        )
        if cos_sim < 0.99:
            return False
    return True


def benchmark_nki(nki_func):
    """Latency benchmark using nki.benchmark (monkey-patched by trn_eval.py)."""
    device = xm.xla_device()
    N = 128
    D = 128

    np.random.seed(42)
    a_np = (np.random.randn(N, N, D) * 0.1).astype(np.float32)
    b_np = (np.random.randn(N, N, D) * 0.1).astype(np.float32)

    a_t = torch.tensor(a_np, dtype=torch.bfloat16, device=device)
    b_t = torch.tensor(b_np, dtype=torch.bfloat16, device=device)

    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(a_t, b_t)
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
