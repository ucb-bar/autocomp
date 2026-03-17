import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

@nki.jit
def test_ncmatmul(q, k, v, kernel_dtype, acc_type, seq_len=2048, d_head=128):
    """Full attention (no masking) using nisa.nc_matmul with ASSIGNMENT (not accumulation)."""
    B_P_SIZE = 128; B_F_SIZE = 512
    LARGE_TILE_SZ = seq_len; REDUCTION_TILE = min(2048, LARGE_TILE_SZ // 2)
    num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE
    num_q_tiles = seq_len // B_P_SIZE

    o = nl.ndarray((seq_len, d_head), dtype=kernel_dtype, buffer=nl.shared_hbm)
    l = nl.ndarray((seq_len, 1), dtype=kernel_dtype, buffer=nl.shared_hbm)
    m = nl.ndarray((seq_len, 1), dtype=kernel_dtype, buffer=nl.shared_hbm)

    for q_tile_idx in nl.affine_range(num_q_tiles):
        q_start = q_tile_idx * B_P_SIZE

        # Load q tile and transpose: q_tile (B_P_SIZE, d_head) -> q_tile_T (d_head, B_P_SIZE)
        q_tile = nl.ndarray((nl.par_dim(B_P_SIZE), d_head), dtype=kernel_dtype)
        q_tile[:, :] = nl.load(q[q_start:q_start+B_P_SIZE, :], dtype=kernel_dtype)

        q_tile_T_psum = nisa.nc_transpose(q_tile)  # (d_head, B_P_SIZE) in psum
        q_tile_T = nl.ndarray((nl.par_dim(d_head), B_P_SIZE), dtype=kernel_dtype)
        q_tile_T[:, :] = nisa.tensor_copy(q_tile_T_psum, dtype=kernel_dtype)

        qk_res_buf = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), buffer=nl.sbuf, dtype=acc_type)
        max_local = nl.ndarray((nl.par_dim(B_P_SIZE), num_k_tile_per_large_tile), dtype=acc_type)

        for k_i in nl.affine_range(num_k_tile_per_large_tile):
            k_start = k_i * B_F_SIZE
            k_tile = nl.ndarray((nl.par_dim(d_head), B_F_SIZE), dtype=kernel_dtype)
            k_tile[:, :] = nl.load(k[:, k_start:k_start+B_F_SIZE], dtype=kernel_dtype)

            # Use nc_matmul with ASSIGNMENT (not += accumulation)
            qk_psum = nl.ndarray((nl.par_dim(B_P_SIZE), B_F_SIZE), dtype=np.float32, buffer=nl.psum)
            qk_psum[:, :] = nisa.nc_matmul(stationary=q_tile_T, moving=k_tile)

            # Copy psum -> sbuf
            qk_sbuf = nl.ndarray((nl.par_dim(B_P_SIZE), B_F_SIZE), dtype=acc_type)
            qk_sbuf[:, :] = nisa.tensor_copy(qk_psum, dtype=acc_type)

            # Store full QK (no masking)
            qk_res_buf[:, k_start:k_start+B_F_SIZE] = nl.copy(qk_sbuf[:, :], dtype=kernel_dtype)
            max_local[:, k_i] = nisa.tensor_reduce(
                np.max, qk_res_buf[:, k_start:k_start+B_F_SIZE], axis=(1,),
                dtype=acc_type, negate=False)

        max_ = nisa.tensor_reduce(np.max, max_local[:, :], axis=(1,), dtype=acc_type, negate=False)
        nl.store(m[q_start:q_start+B_P_SIZE, :], value=nl.copy(max_, dtype=kernel_dtype))
        m_current = max_

        p_local = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
        p_partial_sum = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ // REDUCTION_TILE), dtype=acc_type)
        for k_r_i in nl.affine_range(LARGE_TILE_SZ // REDUCTION_TILE):
            p_local[:, k_r_i * REDUCTION_TILE:(k_r_i + 1) * REDUCTION_TILE] = nisa.activation(
                np.exp,
                qk_res_buf[:, k_r_i * REDUCTION_TILE:(k_r_i + 1) * REDUCTION_TILE],
                bias=-1 * m_current, scale=1.0, dtype=kernel_dtype)
            p_partial_sum[:, k_r_i] = nl.sum(
                p_local[:, k_r_i * REDUCTION_TILE:(k_r_i + 1) * REDUCTION_TILE],
                axis=1, dtype=acc_type)

        p_local_transposed = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
        i_f_128 = nl.arange(B_P_SIZE)[None, :]
        for i_p_t in nl.affine_range(LARGE_TILE_SZ // 512):
            p_local_t_tmp = nl.ndarray((nl.par_dim(B_P_SIZE), 512), buffer=nl.psum, dtype=np.float32)
            for i_p_t_local in nl.affine_range(512 // 128):
                p_local_t_tmp[:, i_p_t_local * 128:(i_p_t_local + 1) * 128] = nisa.nc_transpose(
                    p_local[:, i_p_t * 512 + i_p_t_local * B_P_SIZE:i_p_t * 512 + (i_p_t_local + 1) * B_P_SIZE])
            p_local_transposed[:, i_p_t * 512:(i_p_t + 1) * 512] = nl.copy(
                p_local_t_tmp[:, :], dtype=kernel_dtype)

        ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type)
        pv_psum = nl.zeros((nl.par_dim(B_P_SIZE), d_head), dtype=np.float32, buffer=nl.psum)
        for k_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
            v_tile = nl.load(v[k_i * B_P_SIZE:(k_i + 1) * B_P_SIZE, :], dtype=kernel_dtype)
            pv_psum[:, :] += nl.matmul(p_local_transposed[:, k_i * B_P_SIZE:(k_i+1)*B_P_SIZE],
                                       v_tile, transpose_x=True)

        nl.store(o[q_start:q_start+B_P_SIZE, :], value=nl.copy(pv_psum[:, :], dtype=kernel_dtype))
        nl.store(l[q_start:q_start+B_P_SIZE, :], value=nl.add(nl.log(ps), max_))

    return o, l, m


if __name__ == "__main__":
    q = np.load('csa2048_q.npy'); k = np.load('csa2048_k.npy'); v = np.load('csa2048_v.npy')
    o, l, m = test_ncmatmul(q, k, v, kernel_dtype=nl.bfloat16, acc_type=nl.float32, seq_len=2048, d_head=128)

    def load_f32(arr):
        if arr.dtype.kind == 'V' and arr.dtype.itemsize == 2:
            return (arr.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        return arr.astype(np.float32)

    m1 = load_f32(np.array(m)); l1 = load_f32(np.array(l)); o1 = load_f32(np.array(o))

    # Compare against numpy full attention
    q32 = q.astype(np.float32); k32 = k.astype(np.float32); v32 = v.astype(np.float32)
    qk = q32 @ k32
    m_np = qk.max(axis=1, keepdims=True)
    exp_qk = np.exp(qk - m_np); ps_np = exp_qk.sum(axis=1, keepdims=True)
    o_np = exp_qk @ v32 / ps_np

    norm_o1 = o1 / np.exp(l1 - m1)
    print('m rows 0-5:', m1[:5, 0])
    print('m_np rows 0-5:', m_np[:5, 0])
    print('allclose norm_o vs np full-attn (atol=0.01):', np.allclose(norm_o1, o_np, atol=0.01))
    print('max diff:', np.max(np.abs(norm_o1 - o_np)))
