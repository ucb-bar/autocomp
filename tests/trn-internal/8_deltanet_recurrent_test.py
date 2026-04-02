import numpy as np
import math

import nki
import nki.language as nl
import nki.isa as nisa
import torch
from torch_xla.core import xla_model as xm


# ==============================================================================
# CPU reference for DeltaNet recurrent forward
# ==============================================================================


def deltanet_recurrent_cpu(query, key, value, g, beta):
    """Pure-PyTorch sequential DeltaNet recurrence (CPU reference).

    Args:
        query:  (S, 128) float32
        key:    (S, 128) float32
        value:  (S, 128) float32
        g:      (S, 128) float32 -- log-decay (all 128 cols identical)
        beta:   (S, 128) float32 -- write gate (all 128 cols identical)

    Returns:
        output: (S, 128) float32
    """
    seq_len, dim = query.shape
    state = torch.zeros(dim, dim, dtype=torch.float32)
    output = torch.zeros_like(query)

    for t in range(seq_len):
        q_t = query[t]  # (128,)
        k_t = key[t]  # (128,)
        v_t = value[t]  # (128,)
        g_t = g[t, 0]  # scalar (all cols identical)
        beta_t = beta[t, 0]  # scalar

        # Step 1: Decay state
        state = state * torch.exp(g_t)

        # Step 2: Read memory  kv_mem = state^T @ k_t
        kv_mem = state.T @ k_t  # (128,)

        # Step 3: delta = (v_t - kv_mem) * beta_t
        delta = (v_t - kv_mem) * beta_t

        # Step 4: state += outer(k_t, delta)
        state = state + torch.outer(k_t, delta)

        # Step 5: o_t = state^T @ q_t
        output[t] = state.T @ q_t

    return output


# ==============================================================================
# NKI kernel constants
# ==============================================================================

P_MAX = 128


# SUBSTITUTE HERE


@nki.jit
def ref(
    query,
    key,
    value,
    g_in,
    beta_in,
):
    """Reference: identical to test (DeltaNet recurrent forward)."""
    seq_len, dim = query.shape

    output = nl.ndarray((seq_len, dim), dtype=query.dtype, buffer=nl.shared_hbm)

    seq_stride = dim

    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=state, value=0.0)

    for t in nl.sequential_range(seq_len):
        tok_offset = t * seq_stride

        q_t = nl.ndarray((P_MAX, 1), dtype=query.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=q_t,
            src=query.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        k_t = nl.ndarray((P_MAX, 1), dtype=key.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=k_t,
            src=key.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        v_t = nl.ndarray((P_MAX, 1), dtype=value.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=v_t,
            src=value.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        g_t = nl.ndarray((P_MAX, 1), dtype=g_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=g_t,
            src=g_in.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        beta_t = nl.ndarray((P_MAX, 1), dtype=beta_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=beta_t,
            src=beta_in.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        # Step 1: Decay state
        exp_g = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_g, op=nl.exp, data=g_t, bias=None, scale=1.0)

        state_decayed = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=state_decayed,
            data=state,
            op0=nl.multiply,
            operand0=exp_g,
            engine=nisa.vector_engine,
        )
        nisa.tensor_copy(dst=state, src=state_decayed)

        # Step 2: Read memory
        kv_mem_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kv_mem_psum, stationary=state, moving=k_t)
        kv_mem = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kv_mem, src=kv_mem_psum)

        # Step 3: delta
        v_sub = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=v_sub, data1=v_t, data2=kv_mem, op=nl.subtract)

        delta = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=delta,
            data=v_sub,
            op0=nl.multiply,
            operand0=beta_t,
            engine=nisa.vector_engine,
        )

        # Step 4: state += outer(k_t, delta)
        k_row_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=k_row_psum, data=k_t)
        k_row = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k_row, src=k_row_psum)

        delta_row_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=delta_row_psum, data=delta)
        delta_row = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=delta_row, src=delta_row_psum)

        outer_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=outer_psum, stationary=k_row, moving=delta_row)

        outer_prod = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=outer_prod, src=outer_psum)

        state_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=state_new, data1=state, data2=outer_prod, op=nl.add)
        nisa.tensor_copy(dst=state, src=state_new)

        # Step 5: output
        o_t_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=o_t_psum, stationary=state, moving=q_t)
        o_t = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_t, src=o_t_psum)

        nisa.dma_copy(
            dst=output.ap(pattern=[[1, dim]], offset=tok_offset),
            src=o_t,
        )

    return output


def test_nki(ref_func, test_func):
    """Correctness check: compare NKI ref vs test, and both vs CPU reference."""
    device = xm.xla_device()
    S = 16  # Short sequence for fast testing (sequential loop = O(S))
    D = 128

    for seed in range(2):
        np.random.seed(42 + seed)

        # Generate inputs -- small values to avoid state explosion
        q_np = (np.random.randn(S, D) * 0.1).astype(np.float32)
        k_np = (np.random.randn(S, D) * 0.1).astype(np.float32)
        v_np = (np.random.randn(S, D) * 0.1).astype(np.float32)

        # g: negative log-decay (small magnitude for stability)
        g_scalar = np.random.uniform(-0.5, -0.01, size=(S, 1)).astype(np.float32)
        g_np = np.broadcast_to(g_scalar, (S, D)).copy()

        # beta: write gate in (0, 1)
        beta_scalar = (1.0 / (1.0 + np.exp(-np.random.randn(S, 1)))).astype(np.float32)
        beta_np = np.broadcast_to(beta_scalar, (S, D)).copy()

        # CPU reference
        cpu_out = deltanet_recurrent_cpu(
            torch.tensor(q_np),
            torch.tensor(k_np),
            torch.tensor(v_np),
            torch.tensor(g_np),
            torch.tensor(beta_np),
        )

        # NKI tensors on device
        q_t = torch.tensor(q_np, dtype=torch.float32, device=device)
        k_t = torch.tensor(k_np, dtype=torch.float32, device=device)
        v_t = torch.tensor(v_np, dtype=torch.float32, device=device)
        g_t = torch.tensor(g_np, dtype=torch.float32, device=device)
        beta_t = torch.tensor(beta_np, dtype=torch.float32, device=device)

        result_ref = ref_func(q_t, k_t, v_t, g_t, beta_t)
        result_test = test_func(q_t, k_t, v_t, g_t, beta_t)

        ref_out = result_ref.detach().cpu().float().numpy()
        test_out = result_test.detach().cpu().float().numpy()
        cpu_out_np = cpu_out.numpy()

        # NKI ref vs NKI test (should be identical or very close)
        cos_ref_test = np.dot(ref_out.flatten(), test_out.flatten()) / (
            np.linalg.norm(ref_out.flatten()) * np.linalg.norm(test_out.flatten())
            + 1e-12
        )

        # NKI test vs CPU reference
        cos_test_cpu = np.dot(test_out.flatten(), cpu_out_np.flatten()) / (
            np.linalg.norm(test_out.flatten()) * np.linalg.norm(cpu_out_np.flatten())
            + 1e-12
        )

        print(
            f"  seed={42 + seed}: cos(ref,test)={cos_ref_test:.6f}, cos(test,cpu)={cos_test_cpu:.6f}"
        )

        if cos_ref_test < 0.999:
            print(f"  FAIL: NKI ref vs test cosine {cos_ref_test:.6f} < 0.999")
            return False

        if cos_test_cpu < 0.999:
            print(f"  FAIL: NKI test vs CPU cosine {cos_test_cpu:.6f} < 0.999")
            return False

    return True


def benchmark_nki(nki_func):
    """Latency benchmark using nki.benchmark (monkey-patched by trn_eval.py)."""
    device = xm.xla_device()
    S = 16
    D = 128

    np.random.seed(42)
    q_np = (np.random.randn(S, D) * 0.1).astype(np.float32)
    k_np = (np.random.randn(S, D) * 0.1).astype(np.float32)
    v_np = (np.random.randn(S, D) * 0.1).astype(np.float32)
    g_np = np.broadcast_to(
        np.random.uniform(-0.5, -0.01, size=(S, 1)).astype(np.float32), (S, D)
    ).copy()
    beta_np = np.broadcast_to(
        (1.0 / (1.0 + np.exp(-np.random.randn(S, 1)))).astype(np.float32), (S, D)
    ).copy()

    q_t = torch.tensor(q_np, dtype=torch.float32, device=device)
    k_t = torch.tensor(k_np, dtype=torch.float32, device=device)
    v_t = torch.tensor(v_np, dtype=torch.float32, device=device)
    g_t = torch.tensor(g_np, dtype=torch.float32, device=device)
    beta_t = torch.tensor(beta_np, dtype=torch.float32, device=device)

    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(q_t, k_t, v_t, g_t, beta_t)
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
