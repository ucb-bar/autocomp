import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

NEG_INF = -9984.0

@nki.jit
def flash_attention_core(q, k, v,
                         kernel_dtype, acc_type,
                         seq_len=2048,
                         d_head=128):
    B_P_SIZE = 128
    B_F_SIZE = 512
    LARGE_TILE_SZ = seq_len
    REDUCTION_TILE = min(2048, LARGE_TILE_SZ // 2)
    num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE
    num_q_tiles = seq_len // B_P_SIZE

    i_q_p = nl.arange(B_P_SIZE)[:, None]
    i_q_f = nl.arange(B_F_SIZE)[None, :]
    i_d_p = nl.arange(d_head)[:, None]
    i_d_f = nl.arange(d_head)[None, :]
    i_f_128 = nl.arange(B_P_SIZE)[None, :]
    i_f_k_tiles = nl.arange(num_k_tile_per_large_tile)[None, :]

    o = nl.ndarray((seq_len, d_head), dtype=kernel_dtype, buffer=nl.shared_hbm)
    l = nl.ndarray((seq_len, 1), dtype=kernel_dtype, buffer=nl.shared_hbm)
    m = nl.ndarray((seq_len, 1), dtype=kernel_dtype, buffer=nl.shared_hbm)

    for q_tile_idx in nl.affine_range(num_q_tiles):
        q_local_tile = nl.load(q[q_tile_idx * B_P_SIZE + i_q_p, i_d_f], dtype=kernel_dtype)
        forward_mask = q_tile_idx * B_P_SIZE >= 0
        qk_res_buf = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), buffer=nl.sbuf, dtype=acc_type)
        max_local = nl.ndarray((nl.par_dim(B_P_SIZE), num_k_tile_per_large_tile), dtype=acc_type)

        for k_i in nl.affine_range(num_k_tile_per_large_tile):
            k_tile = nl.load(k[i_d_p, k_i * B_F_SIZE + i_q_f], dtype=kernel_dtype)
            qk_psum = nl.zeros((nl.par_dim(B_P_SIZE), B_F_SIZE), dtype=np.float32, buffer=nl.psum)
            multiplication_required_selection = k_i * B_F_SIZE <= q_tile_idx * B_P_SIZE
            qk_psum[i_q_p, i_q_f] += nl.matmul(q_local_tile, k_tile, transpose_x=True,
                                               mask=multiplication_required_selection)

            left_diagonal_selection = q_tile_idx * B_P_SIZE >= (k_i + 1) * B_F_SIZE
            diagonal_and_right_selection = (q_tile_idx * B_P_SIZE < (k_i + 1) * B_F_SIZE) & forward_mask

            q_pos = q_tile_idx * B_P_SIZE + i_q_p
            k_pos = k_i * B_F_SIZE + i_q_f
            pred = q_pos >= k_pos
            qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f] = nisa.affine_select(
                pred=pred,
                on_true_tile=qk_psum[i_q_p, i_q_f], on_false_value=-9984.0, dtype=kernel_dtype,
                mask=diagonal_and_right_selection)
            qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f] = \
                nl.copy(qk_psum[i_q_p, i_q_f], dtype=kernel_dtype, mask=left_diagonal_selection)

            max_local[i_q_p, k_i] = nisa.tensor_reduce(np.max, qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f],
                                                        axis=(1,), dtype=acc_type, negate=False, mask=forward_mask)

        max_ = nisa.tensor_reduce(np.max, max_local[i_q_p, i_f_k_tiles], axis=(1,),
                                  dtype=acc_type, negate=False, mask=forward_mask)
        nl.store(m[q_tile_idx * B_P_SIZE:q_tile_idx * B_P_SIZE + B_P_SIZE, :], value=nl.copy(max_, dtype=kernel_dtype))
        m_current = max_

        p_local = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
        i_r_f = nl.arange(REDUCTION_TILE)[None, :]
        p_partial_sum = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ // REDUCTION_TILE), dtype=acc_type)
        for k_r_i in nl.affine_range(LARGE_TILE_SZ // REDUCTION_TILE):
            p_local[i_q_p, k_r_i * REDUCTION_TILE + i_r_f] = \
                nisa.activation(np.exp,
                                qk_res_buf[i_q_p, k_r_i * REDUCTION_TILE + i_r_f],
                                bias=-1 * m_current, scale=1.0, dtype=kernel_dtype, mask=forward_mask)
            p_partial_sum[i_q_p, k_r_i] = nl.sum(p_local[i_q_p, k_r_i * REDUCTION_TILE + i_r_f],
                                                   axis=1, dtype=acc_type, mask=forward_mask)

        p_local_transposed = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
        for i_p_t in nl.affine_range(LARGE_TILE_SZ // 512):
            p_local_t_tmp = nl.ndarray((nl.par_dim(B_P_SIZE), 512), buffer=nl.psum, dtype=np.float32)
            for i_p_t_local in nl.affine_range(512 // 128):
                p_local_t_tmp[i_q_p, i_p_t_local * 128 + i_f_128] = nisa.nc_transpose(
                    p_local[i_q_p, i_p_t * 512 + i_p_t_local * B_P_SIZE + i_f_128], mask=forward_mask)
            i_f_512 = nl.arange(512)[None, :]
            p_local_transposed[i_q_p, i_p_t * 512 + i_f_512] = nl.copy(p_local_t_tmp[i_q_p, i_f_512],
                                                                          dtype=kernel_dtype, mask=forward_mask)

        ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type, mask=forward_mask)
        pv_psum = nl.zeros((nl.par_dim(B_P_SIZE), d_head), dtype=np.float32, buffer=nl.psum)
        for k_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
            v_tile = nl.load(v[k_i * B_P_SIZE + i_q_p, i_d_f], dtype=kernel_dtype)
            pv_psum[i_q_p, i_d_f] += nl.matmul(p_local_transposed[i_q_p, k_i * B_P_SIZE + i_f_128],
                                                v_tile, transpose_x=True, mask=forward_mask)

        nl.store(o[q_tile_idx * B_P_SIZE:q_tile_idx * B_P_SIZE + B_P_SIZE, :],
                 value=nl.copy(pv_psum[i_q_p, i_d_f], dtype=kernel_dtype))
        nl.store(l[q_tile_idx * B_P_SIZE:q_tile_idx * B_P_SIZE + B_P_SIZE, :],
                 value=nl.add(nl.log(ps), max_))

    return o, l, m


if __name__ == "__main__":
    np.random.seed(42)
    q = np.load('csa2048_q.npy')
    k = np.load('csa2048_k.npy')
    v = np.load('csa2048_v.npy')

    o, l, m = flash_attention_core(q, k, v, kernel_dtype=nl.bfloat16, acc_type=nl.float32, seq_len=2048, d_head=128)

    def load_f32(arr):
        if arr.dtype.kind == 'V' and arr.dtype.itemsize == 2:
            return (arr.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        return arr.astype(np.float32)

    m1 = load_f32(np.array(m))
    l1 = load_f32(np.array(l))
    o1 = load_f32(np.array(o))

    # Expected causal m from numpy
    q32 = q.astype(np.float32); k32 = k.astype(np.float32)
    qk = q32 @ k32
    NEG_INF_NP = -9984.0
    causal = np.full((2048, 2048), NEG_INF_NP, dtype=np.float32)
    for i in range(2048):
        causal[i, :i+1] = 0.0
    qk_causal = qk + causal
    m_np = qk_causal.max(axis=1)

    print('m1 rows 0-9:', m1[:10, 0])
    print('m_np rows 0-9:', m_np[:10])
    m2 = load_f32(np.load('out_beta2_csa2048_m.npy'))
    print('m2 rows 0-9:', m2[:10, 0])

    norm_o1 = o1 / np.exp(l1 - m1)
    norm_o2 = load_f32(np.load('out_beta2_csa2048_o.npy')) / np.exp(load_f32(np.load('out_beta2_csa2048_l.npy')) - m2)
    print('allclose norm_o1 vs norm_o2 (atol=0.01):', np.allclose(norm_o1, norm_o2, atol=0.01, rtol=0.0))
    print('max diff:', np.max(np.abs(norm_o1 - norm_o2)))
