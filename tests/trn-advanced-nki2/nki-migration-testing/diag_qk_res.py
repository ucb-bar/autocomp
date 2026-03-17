"""Diagnostic: check the actual qk_res_buf values to see if writes are correct."""
import numpy as np
import torch
from torch_xla.core import xla_model as xm
import nki
import nki.isa as nisa
import nki.language as nl

NEG_INF = -9984.0

@nki.jit
def test_qk_res(q, k, causal_mask, seq_len=2048, d_head=128):
    """Extract qk_res_buf values for q_tile=0."""
    B_P_SIZE = 128
    B_F_SIZE = 512
    num_k_tiles = seq_len // B_F_SIZE

    # Only process q_tile_idx=0 (rows 0-127)
    qk_out = nl.ndarray((B_P_SIZE, seq_len), dtype=nl.float32, buffer=nl.shared_hbm)

    # Load and transpose Q for tile 0
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

    # Copy qk_res_buf to HBM output (upcast to f32)
    qk_res_f32 = nl.ndarray((B_P_SIZE, seq_len), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qk_res_f32, src=qk_res_buf)
    nisa.dma_copy(dst=qk_out[0:B_P_SIZE, 0:seq_len], src=qk_res_f32)

    return qk_out


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

    result = test_qk_res(q, k, causal_mask, seq_len=seq_len, d_head=128)
    qk_res = result.cpu().numpy()

    # Reference: compute expected qk values for row 6
    q32 = q_np.astype(np.float32)
    k32 = k_np.astype(np.float32)
    qk = q32 @ k32  # (2048, 2048)

    def to_bf16(x):
        u32 = x.astype(np.float32).view(np.uint32)
        u16 = (u32 >> 16).astype(np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32)

    # For q_tile=0 (rows 0-127), expected qk_res_buf values
    for row in [6, 7, 8]:
        causal_mask_row = np.zeros(seq_len, dtype=np.float32)
        causal_mask_row[:row+1] = 1.0
        qk_masked = (qk[row] + 9984) * causal_mask_row - 9984
        qk_ref = to_bf16(qk_masked)

        nki_vals = qk_res[row, :]

        print(f"\nRow {row}:")
        print(f"  qk_ref (cols 0-10): {qk_ref[:row+2]}")
        print(f"  nki_vals (cols 0-10): {nki_vals[:row+2]}")
        print(f"  match at valid positions: {np.allclose(nki_vals[:row+1], qk_ref[:row+1], atol=0.01)}")
        print(f"  match at masked positions (should be -9984): {np.allclose(nki_vals[row+1:], -9984., atol=1.0)}")
        print(f"  nki max in row: {nki_vals.max():.4f}, ref max: {qk_ref.max():.4f}")
