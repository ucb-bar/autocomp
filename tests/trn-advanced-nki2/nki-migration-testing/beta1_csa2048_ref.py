import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


@nki.jit
def ref(q, k, v,
        kernel_dtype, acc_type,
        seq_len=2048,
        d_head=128
        ):
    """
    Flash attention with causal masking.
    Uses nisa.nc_matmul (assignment) for QK to avoid psum stale-state bugs.
    Uses nisa.affine_select for per-element causal masking.
    """
    B_P_SIZE = 128
    B_F_SIZE = 512
    LARGE_TILE_SZ = seq_len
    REDUCTION_TILE = min(2048, LARGE_TILE_SZ // 2)
    num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE
    num_q_tiles = seq_len // B_P_SIZE

    o = nl.ndarray((seq_len, d_head), dtype=kernel_dtype, buffer=nl.shared_hbm)
    l = nl.ndarray((seq_len, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    m = nl.ndarray((seq_len, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    for q_tile_idx in nl.affine_range(num_q_tiles):
        q_local_tile = nl.load(q[q_tile_idx * B_P_SIZE:(q_tile_idx + 1) * B_P_SIZE, :],
                               dtype=kernel_dtype)
        forward_mask = q_tile_idx * B_P_SIZE >= 0

        # Transpose q tile for nc_matmul: (B_P_SIZE, d_head) -> (d_head, B_P_SIZE)
        q_tile_T_psum = nisa.nc_transpose(q_local_tile)
        q_tile_T = nl.ndarray((nl.par_dim(d_head), B_P_SIZE), dtype=kernel_dtype)
        q_tile_T[:, :] = nisa.tensor_copy(q_tile_T_psum, dtype=kernel_dtype)

        qk_res_buf = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), buffer=nl.sbuf, dtype=acc_type)
        max_local = nl.ndarray((nl.par_dim(B_P_SIZE), num_k_tile_per_large_tile), dtype=acc_type)

        for k_i in nl.affine_range(num_k_tile_per_large_tile):
            k_start = k_i * B_F_SIZE
            k_tile = nl.load(k[:, k_start:k_start + B_F_SIZE], dtype=kernel_dtype)

            # QK via nc_matmul ASSIGNMENT (avoids stale psum accumulation)
            qk_psum = nl.ndarray((nl.par_dim(B_P_SIZE), B_F_SIZE), dtype=np.float32, buffer=nl.psum)
            qk_psum[:, :] = nisa.nc_matmul(stationary=q_tile_T, moving=k_tile)

            left_diagonal_selection = q_tile_idx * B_P_SIZE >= (k_i + 1) * B_F_SIZE
            diagonal_and_right_selection = (q_tile_idx * B_P_SIZE < (k_i + 1) * B_F_SIZE) & forward_mask

            q_pos = q_tile_idx * B_P_SIZE + nl.arange(B_P_SIZE)[:, None]
            k_pos = k_i * B_F_SIZE + nl.arange(B_F_SIZE)[None, :]
            pred = q_pos >= k_pos

            # For diagonal and right tiles: affine_select applies per-element causal mask
            qk_res_buf[:, k_start:k_start + B_F_SIZE] = nisa.affine_select(
                pred=pred,
                on_true_tile=qk_psum[:, :], on_false_value=-9984.0, dtype=kernel_dtype,
                mask=diagonal_and_right_selection)

            # For left-of-diagonal tiles: direct copy (all positions visible)
            qk_res_buf[:, k_start:k_start + B_F_SIZE] = \
                nl.copy(qk_psum[:, :], dtype=kernel_dtype, mask=left_diagonal_selection)

            max_local[:, k_i] = nisa.tensor_reduce(
                np.max, qk_res_buf[:, k_start:k_start + B_F_SIZE], axis=(1,),
                dtype=acc_type, negate=False, mask=forward_mask)

        max_ = nisa.tensor_reduce(np.max, max_local[:, :], axis=(1,),
                                  dtype=acc_type, negate=False, mask=forward_mask)
        nl.store(m[q_tile_idx * B_P_SIZE:q_tile_idx * B_P_SIZE + B_P_SIZE, :],
                 value=nl.copy(max_, dtype=nl.float32))
        m_current = max_

        p_local = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
        p_partial_sum = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ // REDUCTION_TILE), dtype=acc_type)
        for k_r_i in nl.affine_range(LARGE_TILE_SZ // REDUCTION_TILE):
            p_local[:, k_r_i * REDUCTION_TILE:(k_r_i + 1) * REDUCTION_TILE] = \
                nisa.activation(np.exp,
                                qk_res_buf[:, k_r_i * REDUCTION_TILE:(k_r_i + 1) * REDUCTION_TILE],
                                bias=-1 * m_current, scale=1.0, dtype=kernel_dtype,
                                mask=forward_mask)
            p_partial_sum[:, k_r_i] = nl.sum(
                p_local[:, k_r_i * REDUCTION_TILE:(k_r_i + 1) * REDUCTION_TILE],
                axis=1, dtype=acc_type, mask=forward_mask)

        p_local_transposed = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
        for i_p_t in nl.affine_range(LARGE_TILE_SZ // 512):
            p_local_t_tmp = nl.ndarray((nl.par_dim(B_P_SIZE), 512), buffer=nl.psum, dtype=np.float32)
            for i_p_t_local in nl.affine_range(512 // 128):
                p_local_t_tmp[:, i_p_t_local * 128:(i_p_t_local + 1) * 128] = nisa.nc_transpose(
                    p_local[:, i_p_t * 512 + i_p_t_local * B_P_SIZE:i_p_t * 512 + (i_p_t_local + 1) * B_P_SIZE],
                    mask=forward_mask)
            p_local_transposed[:, i_p_t * 512:(i_p_t + 1) * 512] = nl.copy(
                p_local_t_tmp[:, :], dtype=kernel_dtype, mask=forward_mask)

        ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type, mask=forward_mask)
        pv_psum = nl.zeros((nl.par_dim(B_P_SIZE), d_head), dtype=np.float32, buffer=nl.psum)
        for k_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
            v_tile = nl.load(v[k_i * B_P_SIZE:(k_i + 1) * B_P_SIZE, :], dtype=kernel_dtype)
            pv_psum[:, :] += nl.matmul(p_local_transposed[:, k_i * B_P_SIZE:(k_i + 1) * B_P_SIZE],
                                       v_tile, transpose_x=True, mask=forward_mask)

        nl.store(o[q_tile_idx * B_P_SIZE:q_tile_idx * B_P_SIZE + B_P_SIZE, :],
                 value=nl.copy(pv_psum[:, :], dtype=kernel_dtype))
        nl.store(l[q_tile_idx * B_P_SIZE:q_tile_idx * B_P_SIZE + B_P_SIZE, :],
                 value=nl.add(nl.log(ps), max_))

    return o, l, m


if __name__ == "__main__":
    q = np.load("csa2048_q.npy")
    k = np.load("csa2048_k.npy")
    v = np.load("csa2048_v.npy")

    o, l, m = ref(q, k, v, kernel_dtype=nl.bfloat16, acc_type=nl.float32, seq_len=2048, d_head=128)
    np.save("out_beta1_csa2048_o.npy", o)
    np.save("out_beta1_csa2048_l.npy", l)
    np.save("out_beta1_csa2048_m.npy", m)
