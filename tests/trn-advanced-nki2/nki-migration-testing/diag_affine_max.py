"""Same as diag_online_max but using affine_range with conditional."""
import numpy as np
import torch
from torch_xla.core import xla_model as xm
import nki
import nki.isa as nisa
import nki.language as nl

NEG_INF = -9984.0

@nki.jit
def test_affine_max(q, k, causal_mask):
    B_P_SIZE = 128
    B_F_SIZE = 512
    d_head = 128
    seq_len = 2048
    num_k_tiles = seq_len // B_F_SIZE
    q_start = 4 * B_P_SIZE  # qt4 = 512

    m_out = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    q_tile = nl.ndarray((B_P_SIZE, d_head), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_tile, src=q[q_start:q_start+B_P_SIZE, 0:d_head])
    q_tile_T_psum = nl.ndarray((d_head, B_P_SIZE), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=q_tile_T_psum, data=q_tile)
    q_tile_T = nl.ndarray((d_head, B_P_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=q_tile_T, src=q_tile_T_psum)

    running_max = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=running_max, value=NEG_INF)

    k_tile = nl.ndarray((d_head, B_F_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
    causal_tile = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    tile_max = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    new_max = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)

    for k_i in nl.affine_range(num_k_tiles):
        k_start = k_i * B_F_SIZE
        k_end = k_start + B_F_SIZE
        right_of_diag = k_start > q_start + B_P_SIZE - 1

        if not right_of_diag:
            nisa.dma_copy(dst=k_tile, src=k[0:d_head, k_start:k_end])
            qk_psum = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=qk_psum, stationary=q_tile_T, moving=k_tile)
            qk_sbuf = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=qk_sbuf, src=qk_psum)

            left_of_diag = q_start >= k_end
            if left_of_diag:
                masked_qk = qk_sbuf
            else:
                nisa.dma_copy(dst=causal_tile, src=causal_mask[q_start:q_start+B_P_SIZE, k_start:k_end])
                qk_shifted = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=qk_shifted, data=qk_sbuf, op0=nl.add, operand0=-NEG_INF)
                masked_shifted = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=masked_shifted, data1=qk_shifted, data2=causal_tile, op=nl.multiply)
                masked_qk = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=masked_qk, data=masked_shifted, op0=nl.add, operand0=NEG_INF)

            nisa.tensor_reduce(dst=tile_max, op=nl.maximum, data=masked_qk, axis=1, keepdims=True)
            nisa.tensor_tensor(dst=new_max, data1=running_max, data2=tile_max, op=nl.maximum)
            nisa.tensor_copy(dst=running_max, src=new_max)

    nisa.dma_copy(dst=m_out[0:B_P_SIZE, 0:1], src=running_max)
    return m_out


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

    result = test_affine_max(q, k, causal_mask)
    m_out = result.cpu().numpy()

    q_np32 = q_np.astype(np.float32)
    k_np32 = k_np.astype(np.float32)
    qk = q_np32[512:640] @ k_np32
    m_ref = np.array([qk[i, :513+i].max() for i in range(128)])

    print("Affine-range online max for qt4:")
    for r in [42, 53]:
        print(f"  row {512+r} (idx {r}): numpy={m_ref[r]:.4f}, nki={m_out[r, 0]:.4f}")
    print(f"  allclose (atol=0.5): {np.allclose(m_out[:, 0], m_ref, atol=0.5)}")
    print(f"  max diff: {np.max(np.abs(m_out[:, 0] - m_ref)):.4f}")
