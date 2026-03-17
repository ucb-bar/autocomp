"""Check nc_matmul QK values for rows 6-8 in beta2."""
import numpy as np
import torch
from torch_xla.core import xla_model as xm
import nki
import nki.isa as nisa
import nki.language as nl

NEG_INF = -9984.0

@nki.jit
def dump_qk(q, k, seq_len=2048, d_head=128):
    B_P_SIZE = 128
    B_F_SIZE = 512
    num_k_tiles = seq_len // B_F_SIZE

    # Only process first q_tile
    qk_out = nl.ndarray((B_P_SIZE, 16), dtype=nl.float32, buffer=nl.shared_hbm)  # first 16 k positions
    max_out = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    ps_out = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    q_tile = nl.ndarray((B_P_SIZE, d_head), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_tile, src=q[0:B_P_SIZE, 0:d_head])
    q_tile_T_psum = nl.ndarray((d_head, B_P_SIZE), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=q_tile_T_psum, data=q_tile)
    q_tile_T = nl.ndarray((d_head, B_P_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=q_tile_T, src=q_tile_T_psum)

    qk_res_buf = nl.ndarray((B_P_SIZE, seq_len), buffer=nl.sbuf, dtype=nl.float32)
    k_tile = nl.ndarray((d_head, B_F_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
    causal_tile = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)

    # For simplicity, just do first k tile
    nisa.dma_copy(dst=k_tile, src=k[0:d_head, 0:B_F_SIZE])
    qk_psum = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=qk_psum, stationary=q_tile_T, moving=k_tile)
    qk_sbuf = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qk_sbuf, src=qk_psum)

    # Save first 16 QK values (positions 0-15, all attended for rows >= 15)
    nisa.dma_copy(dst=qk_out[0:B_P_SIZE, 0:16], src=qk_sbuf[0:B_P_SIZE, 0:16])

    # Compute full masked qk_res_buf for max/ps
    for k_i in nl.affine_range(num_k_tiles):
        k_start = k_i * B_F_SIZE
        k_end = k_start + B_F_SIZE
        nisa.dma_copy(dst=k_tile, src=k[0:d_head, k_start:k_end])
        qk_psum2 = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=qk_psum2, stationary=q_tile_T, moving=k_tile)
        qk_sbuf2 = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qk_sbuf2, src=qk_psum2)
        # No masking for simplicity - just take raw QK
        nisa.tensor_copy(dst=qk_res_buf[0:B_P_SIZE, k_start:k_end], src=qk_sbuf2)

    max_ = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=max_, op=nl.maximum, data=qk_res_buf, axis=1, keepdims=True)
    nisa.dma_copy(dst=max_out[0:B_P_SIZE, 0:1], src=max_)

    return qk_out, max_out

if __name__ == "__main__":
    q_np = np.load("csa2048_q.npy")
    k_np = np.load("csa2048_k.npy")

    device = xm.xla_device()
    q = torch.from_numpy(q_np).to(device)
    k = torch.from_numpy(k_np).to(device)

    qk_out, max_out = dump_qk(q, k)
    qk = qk_out.cpu().numpy()
    max_ = max_out.cpu().numpy()

    for row in [6, 7, 8, 14]:
        print(f"Row {row}: max_unmasked={max_[row,0]:.6f}, first 16 QK={qk[row, :16]}")
        print(f"  first 10: {qk[row, :10]}")

    # Also compute numpy bf16 matmul
    def to_bf16(x):
        u32 = x.astype(np.float32).view(np.uint32)
        u16 = (u32 >> 16).astype(np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32)

    q32 = to_bf16(q_np[:128])
    k32 = to_bf16(k_np)
    qk_np = q32 @ k32
    print("\nNumpy bf16 matmul (row 8 first 16):", qk_np[8, :16])
    print("Numpy max row 8:", qk_np[8, :9].max(), "(unmasked, first 9 only)")
