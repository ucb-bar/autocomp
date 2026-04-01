import numpy as np
import math

import nki
import nki.language as nl
import nki.isa as nisa
import torch
from torch_xla.core import xla_model as xm


# Tile size constants
PMAX = 128
K_TILE = 512
V_TILE = 128
LARGE_NEG = -9.984e3


# SUBSTITUTE HERE


@nki.jit
def test(q, k, v, scale: float = 1.0):
    """Flash attention with fp32 softmax (non-causal).

    Args:
        q: (batch, seqlen_q, d_head) bf16
        k: (batch, d_head, seqlen_kv) bf16 (transposed)
        v: (batch, seqlen_kv, d_head) bf16
        scale: attention scale factor (typically 1/sqrt(d_head))

    Returns:
        output: (batch, seqlen_q, d_head) bf16
    """
    batch, seqlen_q, d_head = q.shape
    _, _, seqlen_kv = k.shape

    assert d_head == PMAX
    assert seqlen_q % PMAX == 0
    assert seqlen_kv % K_TILE == 0

    n_q_tiles = seqlen_q // PMAX
    n_kv_chunks = seqlen_kv // K_TILE
    n_v_tiles_per_chunk = K_TILE // V_TILE

    output = nl.ndarray((batch, seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    for b in nl.affine_range(batch):
        for q_tile_idx in nl.affine_range(n_q_tiles):
            q_offset = q_tile_idx * PMAX

            q_tile = nl.ndarray((PMAX, d_head), dtype=q.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_tile, src=q[b, nl.ds(q_offset, PMAX), :])

            q_scaled = nl.ndarray((PMAX, d_head), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(q_scaled, q_tile, op0=nl.multiply, operand0=scale)

            running_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(running_max, LARGE_NEG)

            running_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(running_sum, 0.0)

            out_acc = nl.ndarray((PMAX, d_head), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(out_acc, 0.0)

            for kv_chunk_idx in range(n_kv_chunks):
                kv_offset = kv_chunk_idx * K_TILE

                k_tile = nl.ndarray((d_head, K_TILE), dtype=k.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=k_tile, src=k[b, :, nl.ds(kv_offset, K_TILE)])

                q_scaled_bf16 = nl.ndarray(
                    (PMAX, d_head), dtype=nl.bfloat16, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=q_scaled_bf16, src=q_scaled)

                qk_psum = nl.ndarray((PMAX, K_TILE), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(qk_psum, q_scaled_bf16, k_tile)

                qk_sbuf = nl.ndarray((PMAX, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=qk_sbuf, src=qk_psum)

                chunk_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(chunk_max, nl.maximum, qk_sbuf, axis=1)

                new_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(new_max, running_max, chunk_max, op=nl.maximum)

                max_diff = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(max_diff, running_max, new_max, op=nl.subtract)
                correction = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(correction, nl.exp, max_diff)

                nisa.tensor_scalar(
                    out_acc, out_acc, op0=nl.multiply, operand0=correction
                )

                nisa.tensor_scalar(
                    running_sum, running_sum, op0=nl.multiply, operand0=correction
                )

                qk_centered = nl.ndarray(
                    (PMAX, K_TILE), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_scalar(
                    qk_centered, qk_sbuf, op0=nl.subtract, operand0=new_max
                )

                exp_scores = nl.ndarray(
                    (PMAX, K_TILE), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.activation(exp_scores, nl.exp, qk_centered)

                chunk_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(chunk_sum, nl.add, exp_scores, axis=1)
                nisa.tensor_tensor(running_sum, running_sum, chunk_sum, op=nl.add)

                nisa.tensor_copy(dst=running_max, src=new_max)

                pv_psum = nl.ndarray((PMAX, d_head), dtype=nl.float32, buffer=nl.psum)

                for v_tile_idx in nl.affine_range(n_v_tiles_per_chunk):
                    v_offset = kv_offset + v_tile_idx * V_TILE

                    v_tile_bf16 = nl.ndarray(
                        (V_TILE, d_head), dtype=v.dtype, buffer=nl.sbuf
                    )
                    nisa.dma_copy(dst=v_tile_bf16, src=v[b, nl.ds(v_offset, V_TILE), :])

                    exp_chunk_offset = v_tile_idx * V_TILE
                    exp_chunk_bf16 = nl.ndarray(
                        (PMAX, V_TILE), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(
                        dst=exp_chunk_bf16,
                        src=exp_scores[:, nl.ds(exp_chunk_offset, V_TILE)],
                    )

                    nisa.nc_matmul(pv_psum, exp_chunk_bf16, v_tile_bf16)

                pv_sbuf = nl.ndarray((PMAX, d_head), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pv_sbuf, src=pv_psum)

                nisa.tensor_tensor(out_acc, out_acc, pv_sbuf, op=nl.add)

            inv_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(inv_sum, nl.reciprocal, running_sum)

            nisa.tensor_scalar(out_acc, out_acc, op0=nl.multiply, operand0=inv_sum)

            out_tile = nl.ndarray((PMAX, d_head), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_tile, src=out_acc)

            nisa.dma_copy(dst=output[b, nl.ds(q_offset, PMAX), :], src=out_tile)

    return output


@nki.jit
def ref(q, k, v, scale: float = 1.0):
    """Reference: identical to test (FP32 flash attention)."""
    batch, seqlen_q, d_head = q.shape
    _, _, seqlen_kv = k.shape

    assert d_head == PMAX
    assert seqlen_q % PMAX == 0
    assert seqlen_kv % K_TILE == 0

    n_q_tiles = seqlen_q // PMAX
    n_kv_chunks = seqlen_kv // K_TILE
    n_v_tiles_per_chunk = K_TILE // V_TILE

    output = nl.ndarray((batch, seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    for b in nl.affine_range(batch):
        for q_tile_idx in nl.affine_range(n_q_tiles):
            q_offset = q_tile_idx * PMAX

            q_tile = nl.ndarray((PMAX, d_head), dtype=q.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_tile, src=q[b, nl.ds(q_offset, PMAX), :])

            q_scaled = nl.ndarray((PMAX, d_head), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(q_scaled, q_tile, op0=nl.multiply, operand0=scale)

            running_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(running_max, LARGE_NEG)

            running_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(running_sum, 0.0)

            out_acc = nl.ndarray((PMAX, d_head), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(out_acc, 0.0)

            for kv_chunk_idx in range(n_kv_chunks):
                kv_offset = kv_chunk_idx * K_TILE

                k_tile = nl.ndarray((d_head, K_TILE), dtype=k.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=k_tile, src=k[b, :, nl.ds(kv_offset, K_TILE)])

                q_scaled_bf16 = nl.ndarray(
                    (PMAX, d_head), dtype=nl.bfloat16, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=q_scaled_bf16, src=q_scaled)

                qk_psum = nl.ndarray((PMAX, K_TILE), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(qk_psum, q_scaled_bf16, k_tile)

                qk_sbuf = nl.ndarray((PMAX, K_TILE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=qk_sbuf, src=qk_psum)

                chunk_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(chunk_max, nl.maximum, qk_sbuf, axis=1)

                new_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(new_max, running_max, chunk_max, op=nl.maximum)

                max_diff = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(max_diff, running_max, new_max, op=nl.subtract)
                correction = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(correction, nl.exp, max_diff)

                nisa.tensor_scalar(
                    out_acc, out_acc, op0=nl.multiply, operand0=correction
                )

                nisa.tensor_scalar(
                    running_sum, running_sum, op0=nl.multiply, operand0=correction
                )

                qk_centered = nl.ndarray(
                    (PMAX, K_TILE), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_scalar(
                    qk_centered, qk_sbuf, op0=nl.subtract, operand0=new_max
                )

                exp_scores = nl.ndarray(
                    (PMAX, K_TILE), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.activation(exp_scores, nl.exp, qk_centered)

                chunk_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(chunk_sum, nl.add, exp_scores, axis=1)
                nisa.tensor_tensor(running_sum, running_sum, chunk_sum, op=nl.add)

                nisa.tensor_copy(dst=running_max, src=new_max)

                pv_psum = nl.ndarray((PMAX, d_head), dtype=nl.float32, buffer=nl.psum)

                for v_tile_idx in nl.affine_range(n_v_tiles_per_chunk):
                    v_offset = kv_offset + v_tile_idx * V_TILE

                    v_tile_bf16 = nl.ndarray(
                        (V_TILE, d_head), dtype=v.dtype, buffer=nl.sbuf
                    )
                    nisa.dma_copy(dst=v_tile_bf16, src=v[b, nl.ds(v_offset, V_TILE), :])

                    exp_chunk_offset = v_tile_idx * V_TILE
                    exp_chunk_bf16 = nl.ndarray(
                        (PMAX, V_TILE), dtype=nl.bfloat16, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(
                        dst=exp_chunk_bf16,
                        src=exp_scores[:, nl.ds(exp_chunk_offset, V_TILE)],
                    )

                    nisa.nc_matmul(pv_psum, exp_chunk_bf16, v_tile_bf16)

                pv_sbuf = nl.ndarray((PMAX, d_head), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pv_sbuf, src=pv_psum)

                nisa.tensor_tensor(out_acc, out_acc, pv_sbuf, op=nl.add)

            inv_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(inv_sum, nl.reciprocal, running_sum)

            nisa.tensor_scalar(out_acc, out_acc, op0=nl.multiply, operand0=inv_sum)

            out_tile = nl.ndarray((PMAX, d_head), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_tile, src=out_acc)

            nisa.dma_copy(dst=output[b, nl.ds(q_offset, PMAX), :], src=out_tile)

    return output


def reference_attention_cpu(q, k, v, scale):
    """CPU reference: standard attention without masking.

    Args:
        q: (batch, seqlen_q, d_head) float32
        k: (batch, d_head, seqlen_kv) float32
        v: (batch, seqlen_kv, d_head) float32
        scale: float

    Returns:
        output: (batch, seqlen_q, d_head) float32
    """
    import torch.nn.functional as F

    # QK: (batch, seqlen_q, seqlen_kv)
    qk = torch.bmm(q * scale, k)
    attn = F.softmax(qk, dim=-1)
    return torch.bmm(attn, v)


def test_nki(ref_func, test_func):
    """Correctness check: compare ref and test vs CPU attention."""
    device = xm.xla_device()

    batch = 1
    seqlen_q = 512  # 4 Q tiles
    seqlen_kv = 512  # 1 KV chunk
    d_head = 128
    scale = 1.0 / math.sqrt(d_head)

    for seed in range(2):
        np.random.seed(42 + seed)
        q_np = (np.random.randn(batch, seqlen_q, d_head) * 0.1).astype(np.float32)
        # K is transposed: (batch, d_head, seqlen_kv)
        k_np = (np.random.randn(batch, d_head, seqlen_kv) * 0.1).astype(np.float32)
        v_np = (np.random.randn(batch, seqlen_kv, d_head) * 0.1).astype(np.float32)

        # CPU reference
        cpu_out = reference_attention_cpu(
            torch.tensor(q_np), torch.tensor(k_np), torch.tensor(v_np), scale
        ).numpy()

        # NKI on device (bf16 inputs)
        q_dev = torch.tensor(q_np, dtype=torch.bfloat16, device=device)
        k_dev = torch.tensor(k_np, dtype=torch.bfloat16, device=device)
        v_dev = torch.tensor(v_np, dtype=torch.bfloat16, device=device)

        result_ref = ref_func(q_dev, k_dev, v_dev, scale)
        result_test = test_func(q_dev, k_dev, v_dev, scale)

        ref_out = result_ref.detach().cpu().float().numpy()
        test_out = result_test.detach().cpu().float().numpy()

        cos_ref_test = np.dot(ref_out.flatten(), test_out.flatten()) / (
            np.linalg.norm(ref_out.flatten()) * np.linalg.norm(test_out.flatten())
            + 1e-12
        )

        cos_test_cpu = np.dot(test_out.flatten(), cpu_out.flatten()) / (
            np.linalg.norm(test_out.flatten()) * np.linalg.norm(cpu_out.flatten())
            + 1e-12
        )

        print(
            f"  seed={42 + seed}: cos(ref,test)={cos_ref_test:.6f}, cos(test,cpu)={cos_test_cpu:.6f}"
        )

        if cos_ref_test < 0.99:
            print(f"  FAIL: NKI ref vs test cosine {cos_ref_test:.6f} < 0.99")
            return False

        # bf16 softmax has some precision loss, 0.99 is reasonable
        if cos_test_cpu < 0.99:
            print(f"  FAIL: NKI test vs CPU cosine {cos_test_cpu:.6f} < 0.99")
            return False

    return True


def benchmark_nki(nki_func):
    """Latency benchmark using nki.benchmark (monkey-patched by trn_eval.py)."""
    device = xm.xla_device()
    batch = 1
    seqlen_q = 512
    seqlen_kv = 512
    d_head = 128
    scale = 1.0 / math.sqrt(d_head)

    np.random.seed(42)
    q_np = (np.random.randn(batch, seqlen_q, d_head) * 0.1).astype(np.float32)
    k_np = (np.random.randn(batch, d_head, seqlen_kv) * 0.1).astype(np.float32)
    v_np = (np.random.randn(batch, seqlen_kv, d_head) * 0.1).astype(np.float32)

    q_dev = torch.tensor(q_np, dtype=torch.bfloat16, device=device)
    k_dev = torch.tensor(k_np, dtype=torch.bfloat16, device=device)
    v_dev = torch.tensor(v_np, dtype=torch.bfloat16, device=device)

    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(q_dev, k_dev, v_dev, scale)
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
