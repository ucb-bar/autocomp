"""Dump ps values for q_tile=0 to check row 8."""
import numpy as np
import torch
from torch_xla.core import xla_model as xm
import nki
import nki.isa as nisa
import nki.language as nl

NEG_INF = -9984.0

@nki.jit
def dump_ps(q, k, causal_mask, seq_len=2048, d_head=128):
    B_P_SIZE = 128
    B_F_SIZE = 512
    REDUCTION_TILE = 1024
    num_k_tiles = seq_len // B_F_SIZE

    ps_out = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    ps0_out = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    ps1_out = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    max_out = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    q_tile = nl.ndarray((B_P_SIZE, d_head), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_tile, src=q[0:B_P_SIZE, 0:d_head])
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
        nisa.dma_copy(dst=causal_tile, src=causal_mask[0:B_P_SIZE, k_start:k_end])
        qk_shifted = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=qk_shifted, data=qk_sbuf, op0=nl.add, operand0=-NEG_INF)
        masked_shifted = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=masked_shifted, data1=qk_shifted, data2=causal_tile, op=nl.multiply)
        masked_qk = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=masked_qk, data=masked_shifted, op0=nl.add, operand0=NEG_INF)
        nisa.tensor_copy(dst=qk_res_buf[0:B_P_SIZE, k_start:k_end], src=masked_qk)

    max_ = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=max_, op=nl.maximum, data=qk_res_buf, axis=1, keepdims=True)
    nisa.dma_copy(dst=max_out[0:B_P_SIZE, 0:1], src=max_)

    neg_max = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=neg_max, data=max_, op0=nl.multiply, operand0=-1.0)

    p_local_0 = nl.ndarray((B_P_SIZE, REDUCTION_TILE), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=p_local_0, op=nl.exp,
                    data=qk_res_buf[0:B_P_SIZE, 0:1024], bias=neg_max, scale=1.0)
    ps_0 = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ps_0, op=nl.add, data=p_local_0, axis=1, keepdims=True)
    nisa.dma_copy(dst=ps0_out[0:B_P_SIZE, 0:1], src=ps_0)

    p_local_1 = nl.ndarray((B_P_SIZE, REDUCTION_TILE), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=p_local_1, op=nl.exp,
                    data=qk_res_buf[0:B_P_SIZE, 1024:2048], bias=neg_max, scale=1.0)
    ps_1 = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ps_1, op=nl.add, data=p_local_1, axis=1, keepdims=True)
    nisa.dma_copy(dst=ps1_out[0:B_P_SIZE, 0:1], src=ps_1)

    ps = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ps, data1=ps_0, data2=ps_1, op=nl.add)
    nisa.dma_copy(dst=ps_out[0:B_P_SIZE, 0:1], src=ps)

    return ps_out, ps0_out, ps1_out, max_out

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

    ps_out, ps0_out, ps1_out, max_out = dump_ps(q, k, causal_mask)
    ps = ps_out.cpu().numpy()
    ps0 = ps0_out.cpu().numpy()
    ps1 = ps1_out.cpu().numpy()
    max_ = max_out.cpu().numpy()

    for row in [6, 7, 8, 14]:
        print(f"Row {row}: max={max_[row,0]:.6f}, ps0={ps0[row,0]:.4f}, ps1={ps1[row,0]:.4f}, ps={ps[row,0]:.4f}")
        import math
        l_f32 = math.log(ps[row,0]) + max_[row,0]
        print(f"  log(ps)+max = {l_f32:.6f} -> bf16 = {'%.4f' % round(l_f32/0.25)*0.25}")

    # Compare with beta1
    def load_f32(path):
        arr = np.load(path)
        if arr.dtype.kind == 'V' and arr.dtype.itemsize == 2:
            return (arr.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        return arr.astype(np.float32)
    l1 = load_f32("out_beta1_csa2048_l.npy")
    m1 = load_f32("out_beta1_csa2048_m.npy")
    print("\nBeta1 reference (first tile rows):")
    for row in [6, 7, 8, 14]:
        print(f"  Row {row}: l1={l1[row,0]:.4f}, m1={m1[row,0]:.4f}")
