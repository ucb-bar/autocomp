"""Dump p_local_0 and p_local_1 for q_tile=0."""
import numpy as np
import torch
from torch_xla.core import xla_model as xm
import nki
import nki.isa as nisa
import nki.language as nl

NEG_INF = -9984.0

@nki.jit
def test_p_dump(q, k, causal_mask, seq_len=2048, d_head=128):
    B_P_SIZE = 128
    B_F_SIZE = 512
    REDUCTION_TILE = 1024
    num_k_tiles = seq_len // B_F_SIZE

    p0_out = nl.ndarray((B_P_SIZE, REDUCTION_TILE), dtype=nl.float32, buffer=nl.shared_hbm)
    p1_out = nl.ndarray((B_P_SIZE, REDUCTION_TILE), dtype=nl.float32, buffer=nl.shared_hbm)
    max_out = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    q_tile = nl.ndarray((B_P_SIZE, d_head), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_tile, src=q[0:B_P_SIZE, 0:d_head])
    q_tile_T_psum = nl.ndarray((d_head, B_P_SIZE), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=q_tile_T_psum, data=q_tile)
    q_tile_T = nl.ndarray((d_head, B_P_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=q_tile_T, src=q_tile_T_psum)

    qk_res_buf = nl.ndarray((B_P_SIZE, seq_len), buffer=nl.sbuf, dtype=nl.bfloat16)
    k_tile = nl.ndarray((d_head, B_F_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
    causal_tile = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)

    for k_i in nl.affine_range(num_k_tiles):
        k_start = k_i * B_F_SIZE
        k_end = k_start + B_F_SIZE
        nisa.dma_copy(dst=k_tile, src=k[0:d_head, k_start:k_end])
        qk_psum = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=qk_psum, stationary=q_tile_T, moving=k_tile)
        qk_sbuf = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qk_sbuf, src=qk_psum)
        nisa.dma_copy(dst=causal_tile, src=causal_mask[0:B_P_SIZE, k_start:k_end])
        qk_shifted = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=qk_shifted, data=qk_sbuf, op0=nl.add, operand0=-NEG_INF)
        masked_shifted = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=masked_shifted, data1=qk_shifted, data2=causal_tile, op=nl.multiply)
        masked_qk = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=masked_qk, data=masked_shifted, op0=nl.add, operand0=NEG_INF)
        masked_qk_bf16 = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=masked_qk_bf16, src=masked_qk)
        nisa.tensor_copy(dst=qk_res_buf[0:B_P_SIZE, k_start:k_end], src=masked_qk_bf16)

    max_ = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=max_, op=nl.maximum, data=qk_res_buf, axis=1, keepdims=True)
    nisa.dma_copy(dst=max_out[0:B_P_SIZE, 0:1], src=max_)

    neg_max = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=neg_max, data=max_, op0=nl.multiply, operand0=-1.0)

    p_local_0 = nl.ndarray((B_P_SIZE, REDUCTION_TILE), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=p_local_0, op=nl.exp,
                    data=qk_res_buf[0:B_P_SIZE, 0:1024], bias=neg_max, scale=1.0)

    p_local_1 = nl.ndarray((B_P_SIZE, REDUCTION_TILE), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=p_local_1, op=nl.exp,
                    data=qk_res_buf[0:B_P_SIZE, 1024:2048], bias=neg_max, scale=1.0)

    # Save p_local_0 and p_local_1 as f32
    p0_f32 = nl.ndarray((B_P_SIZE, REDUCTION_TILE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=p0_f32, src=p_local_0)
    nisa.dma_copy(dst=p0_out[0:B_P_SIZE, 0:REDUCTION_TILE], src=p0_f32)

    p1_f32 = nl.ndarray((B_P_SIZE, REDUCTION_TILE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=p1_f32, src=p_local_1)
    nisa.dma_copy(dst=p1_out[0:B_P_SIZE, 0:REDUCTION_TILE], src=p1_f32)

    return p0_out, p1_out, max_out


if __name__ == "__main__":
    q_np = np.load("csa2048_q.npy")
    k_np = np.load("csa2048_k.npy")
    seq_len = 2048

    causal_mask_np = np.zeros((seq_len, seq_len), dtype=np.float32)
    for i in range(seq_len):
        causal_mask_np[i, :i+1] = 1.0

    device = xm.xla_device()
    q = torch.from_numpy(q_np).to(device)
    k = torch.from_numpy(k_np).to(device)
    causal_mask = torch.from_numpy(causal_mask_np).to(device)

    p0_out, p1_out, max_out = test_p_dump(q, k, causal_mask, seq_len=seq_len, d_head=128)
    p0 = p0_out.cpu().numpy()
    p1 = p1_out.cpu().numpy()
    m = max_out.cpu().numpy()

    # For row 6: expected exp values
    row = 6
    print(f"Row {row}: max_nki={m[row,0]:.4f}")
    ps_nki = p0[row].sum() + p1[row].sum()
    print(f"  p_local_0 sum = {p0[row].sum():.4f}")
    print(f"  p_local_1 sum = {p1[row].sum():.4f}")
    print(f"  total ps = {ps_nki:.4f}")
    print(f"  p_local_0 nonzero entries: {(p0[row] > 1e-10).sum()}")
    print(f"  p_local_1 nonzero entries: {(p1[row] > 1e-10).sum()}")
    print(f"  p_local_0 first 10: {p0[row, :10]}")

    # Reference
    q32 = q_np.astype(np.float32)
    k32 = k_np.astype(np.float32)

    def to_bf16(x):
        u32 = x.astype(np.float32).view(np.uint32)
        u16 = (u32 >> 16).astype(np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32)

    # Use actual NKI qk values from diag_qk_res
    # For reference, compute using bf16 matmul approximation
    q_bf16 = to_bf16(q32[:128])  # first 128 rows
    k_bf16 = to_bf16(k32)  # all k
    qk_bf16_matmul = q_bf16 @ k_bf16  # float32 result of bf16 inputs
    # Mask
    for i in range(128):
        qk_bf16_matmul[i, i+1:] = -9984.0
    qk_bf16_matmul_quantized = to_bf16(qk_bf16_matmul)

    for r in [6, 7, 8]:
        max_ref = qk_bf16_matmul_quantized[r].max()
        ps_ref = np.exp(qk_bf16_matmul_quantized[r] - max_ref).sum()
        print(f"\nRow {r}:")
        print(f"  max_nki={m[r,0]:.4f}, max_ref_bf16mat={max_ref:.4f}")
        print(f"  ps_nki={p0[r].sum() + p1[r].sum():.4f}, ps_ref_bf16mat={ps_ref:.4f}")
