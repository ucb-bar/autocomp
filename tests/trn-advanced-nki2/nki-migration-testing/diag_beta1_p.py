"""Save beta1 p_local values for comparison."""
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

@nki.jit
def ref_p_dump(q, k, causal_mask, seq_len=2048, d_head=128):
    B_P_SIZE = 128
    B_F_SIZE = 512
    LARGE_TILE_SZ = seq_len
    REDUCTION_TILE = min(2048, LARGE_TILE_SZ // 2)
    num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE

    p0_out = nl.ndarray((B_P_SIZE, REDUCTION_TILE), dtype=np.float32, buffer=nl.shared_hbm)
    p1_out = nl.ndarray((B_P_SIZE, REDUCTION_TILE), dtype=np.float32, buffer=nl.shared_hbm)
    max_out = nl.ndarray((B_P_SIZE, 1), dtype=np.float32, buffer=nl.shared_hbm)

    # Only process q_tile_idx=0
    q_local_tile = nl.load(q[0:B_P_SIZE, :], dtype=nl.bfloat16)
    forward_mask = 0 >= 0

    q_tile_T_psum = nisa.nc_transpose(q_local_tile)
    q_tile_T = nl.ndarray((nl.par_dim(d_head), B_P_SIZE), dtype=nl.bfloat16)
    q_tile_T[:, :] = nisa.tensor_copy(q_tile_T_psum, dtype=nl.bfloat16)

    qk_res_buf = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), buffer=nl.sbuf, dtype=np.float32)
    max_local = nl.ndarray((nl.par_dim(B_P_SIZE), num_k_tile_per_large_tile), dtype=np.float32)

    for k_i in nl.affine_range(num_k_tile_per_large_tile):
        k_start = k_i * B_F_SIZE
        k_tile = nl.load(k[:, k_start:k_start + B_F_SIZE], dtype=nl.bfloat16)

        qk_psum = nl.ndarray((nl.par_dim(B_P_SIZE), B_F_SIZE), dtype=np.float32, buffer=nl.psum)
        qk_psum[:, :] = nisa.nc_matmul(stationary=q_tile_T, moving=k_tile)

        q_pos = nl.arange(B_P_SIZE)[:, None]
        k_pos = k_i * B_F_SIZE + nl.arange(B_F_SIZE)[None, :]
        pred = q_pos >= k_pos

        left_diagonal_selection = 0 >= (k_i + 1) * B_F_SIZE
        diagonal_and_right_selection = (0 < (k_i + 1) * B_F_SIZE) & forward_mask

        qk_res_buf[:, k_start:k_start + B_F_SIZE] = nisa.affine_select(
            pred=pred,
            on_true_tile=qk_psum[:, :], on_false_value=-9984.0, dtype=nl.bfloat16,
            mask=diagonal_and_right_selection)

        qk_res_buf[:, k_start:k_start + B_F_SIZE] = \
            nl.copy(qk_psum[:, :], dtype=nl.bfloat16, mask=left_diagonal_selection)

        max_local[:, k_i] = nisa.tensor_reduce(
            np.max, qk_res_buf[:, k_start:k_start + B_F_SIZE], axis=(1,),
            dtype=np.float32, negate=False, mask=forward_mask)

    max_ = nisa.tensor_reduce(np.max, max_local[:, :], axis=(1,),
                              dtype=np.float32, negate=False, mask=forward_mask)
    nl.store(max_out[0:B_P_SIZE, :], value=nl.copy(max_, dtype=np.float32))
    m_current = max_

    p_local = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=nl.bfloat16)
    p_partial_sum = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ // REDUCTION_TILE), dtype=np.float32)
    for k_r_i in nl.affine_range(LARGE_TILE_SZ // REDUCTION_TILE):
        p_local[:, k_r_i * REDUCTION_TILE:(k_r_i + 1) * REDUCTION_TILE] = \
            nisa.activation(np.exp,
                            qk_res_buf[:, k_r_i * REDUCTION_TILE:(k_r_i + 1) * REDUCTION_TILE],
                            bias=-1 * m_current, scale=1.0, dtype=nl.bfloat16,
                            mask=forward_mask)
        p_partial_sum[:, k_r_i] = nl.sum(
            p_local[:, k_r_i * REDUCTION_TILE:(k_r_i + 1) * REDUCTION_TILE],
            axis=1, dtype=np.float32, mask=forward_mask)

    # Save p_local[0:128, 0:1024] and [1024:2048]
    p0_tmp = nl.ndarray((nl.par_dim(B_P_SIZE), REDUCTION_TILE), dtype=np.float32)
    for c in nl.affine_range(REDUCTION_TILE // B_P_SIZE):
        p0_tmp[:, c*B_P_SIZE:(c+1)*B_P_SIZE] = nl.copy(
            p_local[:, c*B_P_SIZE:(c+1)*B_P_SIZE], dtype=np.float32)
    nl.store(p0_out[0:B_P_SIZE, :], value=p0_tmp[:, :])

    p1_tmp = nl.ndarray((nl.par_dim(B_P_SIZE), REDUCTION_TILE), dtype=np.float32)
    for c in nl.affine_range(REDUCTION_TILE // B_P_SIZE):
        p1_tmp[:, c*B_P_SIZE:(c+1)*B_P_SIZE] = nl.copy(
            p_local[:, REDUCTION_TILE + c*B_P_SIZE:REDUCTION_TILE + (c+1)*B_P_SIZE], dtype=np.float32)
    nl.store(p1_out[0:B_P_SIZE, :], value=p1_tmp[:, :])

    return p0_out, p1_out, max_out


if __name__ == "__main__":
    q = np.load("csa2048_q.npy")
    k = np.load("csa2048_k.npy")
    seq_len = 2048

    causal_mask_np = np.zeros((seq_len, seq_len), dtype=np.float32)
    for i in range(seq_len):
        causal_mask_np[i, :i+1] = 1.0

    p0, p1, m = ref_p_dump(q, k, causal_mask_np, seq_len=seq_len, d_head=128)

    for row in [6, 7, 8]:
        ps = p0[row].sum() + p1[row].sum()
        print(f"Row {row}: max={m[row,0]:.4f}, p0_sum={p0[row].sum():.4f}, p1_sum={p1[row].sum():.4f}, total_ps={ps:.4f}")
        print(f"  p0 nonzero: {(p0[row]>1e-10).sum()}, p0 first 10: {p0[row,:10]}")
