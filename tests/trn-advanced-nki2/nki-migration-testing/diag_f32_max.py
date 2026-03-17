"""Check float32 max values in beta2 for specific rows."""
import numpy as np
import torch
from torch_xla.core import xla_model as xm
import nki
import nki.isa as nisa
import nki.language as nl

NEG_INF = -9984.0

@nki.jit
def test_f32_max(q, k, causal_mask, seq_len=2048, d_head=128):
    B_P_SIZE = 128
    B_F_SIZE = 512
    num_k_tiles = seq_len // B_F_SIZE
    num_q_tiles = seq_len // B_P_SIZE

    m_f32_out = nl.ndarray((seq_len, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    for q_tile_idx in nl.sequential_range(num_q_tiles):
        q_start = q_tile_idx * B_P_SIZE
        q_end = q_start + B_P_SIZE

        q_tile = nl.ndarray((B_P_SIZE, d_head), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_tile, src=q[q_start:q_end, 0:d_head])
        q_tile_T_psum = nl.ndarray((d_head, B_P_SIZE), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=q_tile_T_psum, data=q_tile)
        q_tile_T = nl.ndarray((d_head, B_P_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_tile_T, src=q_tile_T_psum)

        qk_res_buf = nl.ndarray((B_P_SIZE, seq_len), buffer=nl.sbuf, dtype=nl.float32)
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
            nisa.dma_copy(dst=causal_tile, src=causal_mask[q_start:q_end, k_start:k_end])
            qk_shifted = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=qk_shifted, data=qk_sbuf, op0=nl.add, operand0=-NEG_INF)
            masked_shifted = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=masked_shifted, data1=qk_shifted, data2=causal_tile, op=nl.multiply)
            masked_qk = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=masked_qk, data=masked_shifted, op0=nl.add, operand0=NEG_INF)
            nisa.tensor_copy(dst=qk_res_buf[0:B_P_SIZE, k_start:k_end], src=masked_qk)

        max_ = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=max_, op=nl.maximum, data=qk_res_buf, axis=1, keepdims=True)
        nisa.dma_copy(dst=m_f32_out[q_start:q_end, 0:1], src=max_)

    return m_f32_out


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

    result = test_f32_max(q, k, causal_mask, seq_len=seq_len, d_head=128)
    m_f32 = result.cpu().numpy()

    # Compare with beta1
    import subprocess
    out_beta1 = np.load("out_beta1_csa2048_m.npy")
    def load_f32(path):
        arr = np.load(path)
        if arr.dtype.kind == 'V' and arr.dtype.itemsize == 2:
            return (arr.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        return arr.astype(np.float32)
    m1 = load_f32("out_beta1_csa2048_m.npy")

    print("Float32 max for specific rows:")
    for row in [6, 8, 14, 1004, 349]:
        print(f"  row {row}: beta2_f32_max={m_f32[row,0]:.6f}, beta1_m={m1[row,0]:.4f}")

    # Check where float32 max differs significantly
    bf16_m2 = np.zeros_like(m_f32)
    for i in range(seq_len):
        v = m_f32[i, 0]
        u32 = np.array([v]).view(np.uint32)
        u16 = (u32 >> 16).astype(np.uint16)
        bf16_m2[i, 0] = (u16.astype(np.uint32) << 16).view(np.float32)[0]

    diff = np.abs(bf16_m2[:, 0] - m1[:, 0])
    print(f"m diff (bf16 of beta2 f32 max vs beta1 m): max={diff.max():.4f}, num>0.1: {(diff>0.1).sum()}")
    bad_rows = np.where(diff > 0.1)[0]
    for r in bad_rows[:10]:
        print(f"  row {r}: beta2_f32={m_f32[r,0]:.6f}, bf16(beta2_f32)={bf16_m2[r,0]:.4f}, beta1_m={m1[r,0]:.4f}")
