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
    q,
    k,
    v,
    bias,
    scale=0.1767766952966369,
):
    """Reference: identical to test (triangular attention, Boltz-2 pairformer)."""
    N, N2, Hd = q.shape
    N_bias, N_bias2, H = bias.shape
    d = Hd // H
    P_MAX = 128

    assert N == N2
    assert N == N_bias and N == N_bias2
    assert N % P_MAX == 0
    assert d <= P_MAX

    output = nl.ndarray((N, N, Hd), dtype=q.dtype, buffer=nl.shared_hbm)

    q_stride_row = N * Hd
    q_stride_col = Hd
    bias_stride_q = N * H
    bias_stride_k = H

    for i_row in nl.affine_range(N):
        row_base = i_row * q_stride_row

        for h in nl.affine_range(H):
            hd_start = h * d

            for j_tile in nl.affine_range(N // P_MAX):
                j_start = j_tile * P_MAX

                q_tile = nl.ndarray((P_MAX, d), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=q_tile,
                    src=q.ap(
                        pattern=[[q_stride_col, P_MAX], [1, d]],
                        offset=row_base + j_start * q_stride_col + hd_start,
                    ),
                )

                q_padded = nl.ndarray((P_MAX, P_MAX), dtype=q.dtype, buffer=nl.sbuf)
                nisa.memset(dst=q_padded, value=0.0)
                nisa.tensor_copy(dst=q_padded[0:P_MAX, 0:d], src=q_tile)

                q_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=q.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=q_t_psum, data=q_padded)
                q_t = nl.ndarray((P_MAX, P_MAX), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=q_t, src=q_t_psum)

                m_prev = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=m_prev, value=-1e30)

                l_prev = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=l_prev, value=0.0)

                o_acc = nl.ndarray((P_MAX, d), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=o_acc, value=0.0)

                for k_tile_idx in nl.sequential_range(N // P_MAX):
                    k_start = k_tile_idx * P_MAX

                    k_tile_sb = nl.ndarray((P_MAX, d), dtype=k.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=k_tile_sb,
                        src=k.ap(
                            pattern=[[q_stride_col, P_MAX], [1, d]],
                            offset=row_base + k_start * q_stride_col + hd_start,
                        ),
                    )

                    v_tile_sb = nl.ndarray((P_MAX, d), dtype=v.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=v_tile_sb,
                        src=v.ap(
                            pattern=[[q_stride_col, P_MAX], [1, d]],
                            offset=row_base + k_start * q_stride_col + hd_start,
                        ),
                    )

                    bias_tile = nl.ndarray(
                        (P_MAX, P_MAX), dtype=bias.dtype, buffer=nl.sbuf
                    )
                    nisa.dma_copy(
                        dst=bias_tile,
                        src=bias.ap(
                            pattern=[[bias_stride_q, P_MAX], [bias_stride_k, P_MAX]],
                            offset=j_start * bias_stride_q
                            + k_start * bias_stride_k
                            + h,
                        ),
                    )

                    k_padded = nl.ndarray((P_MAX, P_MAX), dtype=k.dtype, buffer=nl.sbuf)
                    nisa.memset(dst=k_padded, value=0.0)
                    nisa.tensor_copy(dst=k_padded[0:P_MAX, 0:d], src=k_tile_sb)

                    k_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=k.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=k_t_psum, data=k_padded)
                    k_t = nl.ndarray((P_MAX, P_MAX), dtype=k.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=k_t, src=k_t_psum)

                    logits_psum = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum
                    )
                    nisa.nc_matmul(dst=logits_psum, stationary=q_t, moving=k_t)

                    logits = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=logits, src=logits_psum)

                    logits_scaled = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_scalar(
                        dst=logits_scaled,
                        data=logits,
                        op0=nl.multiply,
                        operand0=scale,
                        engine=nisa.vector_engine,
                    )

                    bias_fp32 = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(dst=bias_fp32, src=bias_tile)

                    logits_biased = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_tensor(
                        dst=logits_biased,
                        data1=logits_scaled,
                        data2=bias_fp32,
                        op=nl.add,
                    )

                    tile_max = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_reduce(
                        dst=tile_max, op=nl.maximum, data=logits_biased, axis=1
                    )

                    m_new = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=m_new, data1=m_prev, data2=tile_max, op=nl.maximum
                    )

                    m_diff = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=m_diff, data1=m_prev, data2=m_new, op=nl.subtract
                    )
                    correction = nl.ndarray(
                        (P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.activation(
                        dst=correction, op=nl.exp, data=m_diff, bias=None, scale=1.0
                    )

                    logits_shifted = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_scalar(
                        dst=logits_shifted,
                        data=logits_biased,
                        op0=nl.subtract,
                        operand0=m_new,
                        engine=nisa.vector_engine,
                    )
                    exp_logits = nl.ndarray(
                        (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.activation(
                        dst=exp_logits,
                        op=nl.exp,
                        data=logits_shifted,
                        bias=None,
                        scale=1.0,
                    )

                    l_corrected = nl.ndarray(
                        (P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_tensor(
                        dst=l_corrected, data1=l_prev, data2=correction, op=nl.multiply
                    )
                    tile_sum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_reduce(dst=tile_sum, op=nl.add, data=exp_logits, axis=1)
                    l_new = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=l_new, data1=l_corrected, data2=tile_sum, op=nl.add
                    )

                    o_scaled = nl.ndarray((P_MAX, d), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(
                        dst=o_scaled,
                        data=o_acc,
                        op0=nl.multiply,
                        operand0=correction,
                        engine=nisa.vector_engine,
                    )

                    exp_bf16 = nl.ndarray((P_MAX, P_MAX), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=exp_bf16, src=exp_logits)

                    exp_t_psum = nl.ndarray(
                        (P_MAX, P_MAX), dtype=q.dtype, buffer=nl.psum
                    )
                    nisa.nc_transpose(dst=exp_t_psum, data=exp_bf16)
                    exp_t = nl.ndarray((P_MAX, P_MAX), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=exp_t, src=exp_t_psum)

                    pv_psum = nl.ndarray((P_MAX, d), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=pv_psum, stationary=exp_t, moving=v_tile_sb)

                    pv_sbuf = nl.ndarray((P_MAX, d), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=pv_sbuf, src=pv_psum)

                    nisa.tensor_tensor(
                        dst=o_acc, data1=o_scaled, data2=pv_sbuf, op=nl.add
                    )

                    nisa.tensor_copy(dst=m_prev, src=m_new)
                    nisa.tensor_copy(dst=l_prev, src=l_new)

                inv_l = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.reciprocal(dst=inv_l, data=l_prev)

                o_final = nl.ndarray((P_MAX, d), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=o_final,
                    data=o_acc,
                    op0=nl.multiply,
                    operand0=inv_l,
                    engine=nisa.vector_engine,
                )

                o_out = nl.ndarray((P_MAX, d), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=o_out, src=o_final)

                nisa.dma_copy(
                    dst=output.ap(
                        pattern=[[q_stride_col, P_MAX], [1, d]],
                        offset=row_base + j_start * q_stride_col + hd_start,
                    ),
                    src=o_out,
                )

    return output


def test_nki(ref_func, test_func):
    """Correctness check: compare ref and test on device tensors."""
    device = xm.xla_device()
    N = 128
    H = 4
    d = 32
    Hd = H * d
    scale = 1.0 / math.sqrt(d)

    for seed in range(2):
        np.random.seed(42 + seed)
        q_np = (np.random.randn(N, N, Hd) * 0.1).astype(np.float32)
        k_np = (np.random.randn(N, N, Hd) * 0.1).astype(np.float32)
        v_np = (np.random.randn(N, N, Hd) * 0.1).astype(np.float32)
        bias_np = (np.random.randn(N, N, H) * 0.1).astype(np.float32)

        q_t = torch.tensor(q_np, dtype=torch.bfloat16, device=device)
        k_t = torch.tensor(k_np, dtype=torch.bfloat16, device=device)
        v_t = torch.tensor(v_np, dtype=torch.bfloat16, device=device)
        bias_t = torch.tensor(bias_np, dtype=torch.bfloat16, device=device)

        result_ref = ref_func(q_t, k_t, v_t, bias_t, scale)
        result_test = test_func(q_t, k_t, v_t, bias_t, scale)

        ref_np = result_ref.detach().cpu().float().numpy()
        test_np = result_test.detach().cpu().float().numpy()

        # Cosine similarity check (bf16 matmul has limited precision)
        cos_sim = np.dot(ref_np.flatten(), test_np.flatten()) / (
            np.linalg.norm(ref_np.flatten()) * np.linalg.norm(test_np.flatten()) + 1e-12
        )
        if cos_sim < 0.999:
            return False
    return True


def benchmark_nki(nki_func):
    """Latency benchmark using nki.benchmark (monkey-patched by trn_eval.py)."""
    device = xm.xla_device()
    N = 128
    H = 4
    d = 32
    Hd = H * d
    scale = 1.0 / math.sqrt(d)

    np.random.seed(42)
    q_np = (np.random.randn(N, N, Hd) * 0.1).astype(np.float32)
    k_np = (np.random.randn(N, N, Hd) * 0.1).astype(np.float32)
    v_np = (np.random.randn(N, N, Hd) * 0.1).astype(np.float32)
    bias_np = (np.random.randn(N, N, H) * 0.1).astype(np.float32)

    q_t = torch.tensor(q_np, dtype=torch.bfloat16, device=device)
    k_t = torch.tensor(k_np, dtype=torch.bfloat16, device=device)
    v_t = torch.tensor(v_np, dtype=torch.bfloat16, device=device)
    bias_t = torch.tensor(bias_np, dtype=torch.bfloat16, device=device)

    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(q_t, k_t, v_t, bias_t, scale)
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
