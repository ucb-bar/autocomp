@nki.jit
def test(q_ref, k_ref, v_ref, use_causal_mask=False, mixed_precision=True):
    # Use q_ref dtype as the intermediate tensor dtype
    kernel_dtype = q_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
    assert q_ref.dtype == k_ref.dtype == v_ref.dtype

    # Shape checking
    seqlen, d_head = q_ref.shape
    assert d_head <= 128, "Cannot use this kernel for d_head > 128"
    assert tuple(k_ref.shape) == (seqlen, d_head)
    assert tuple(v_ref.shape) == (seqlen, d_head)
    out_ref = nl.ndarray((seqlen, d_head), dtype=q_ref.dtype,
                         buffer=nl.shared_hbm)

    softmax_scale = 0.125

    # tile counts and sizes
    q_seq_n_tiles = seqlen // 128
    k_seq_n_tiles = seqlen // 128
    v_seq_n_tiles = seqlen // 128
    q_seq_tile_size = 128
    k_seq_tile_size = 128
    v_seq_tile_size = 128
    d_head_tile_size = d_head

    # Hoist index tensors
    ip_q      = nl.arange(d_head_tile_size)[:, None]
    if_q      = nl.arange(q_seq_tile_size)[None, :]
    ip_k      = nl.arange(d_head_tile_size)[:, None]
    if_k      = nl.arange(k_seq_tile_size)[None, :]
    v_rows    = nl.arange(v_seq_tile_size)[:, None]
    head_cols = nl.arange(d_head_tile_size)[None, :]
    seq_rows  = nl.arange(q_seq_tile_size)[:, None]

    ip_qk     = nl.arange(q_seq_tile_size)[:, None]
    if_qk     = nl.arange(k_seq_tile_size)[None, :]
    ip_max    = nl.arange(q_seq_tile_size)[:, None]
    if_max    = nl.arange(k_seq_n_tiles)[None, :]
    ip_softmax= nl.arange(q_seq_tile_size)[:, None]
    if_softmax= nl.arange(seqlen)[None, :]
    ip_scores = nl.arange(q_seq_tile_size)[:, None]
    if_scores = nl.arange(k_seq_tile_size)[None, :]
    ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
    if_scores_t = nl.arange(q_seq_tile_size)[None, :]
    ip_v_t    = nl.arange(k_seq_tile_size)[:, None]
    if_v_t    = nl.arange(d_head_tile_size)[None, :]
    ip_out    = nl.arange(d_head_tile_size)[:, None]
    if_out    = nl.arange(q_seq_tile_size)[None, :]

    # Step 1: preload and transpose Q, K, V into SBUF in pe_in_dt
    trans_v = nl.ndarray(
        (nl.par_dim(v_seq_tile_size), v_seq_n_tiles, d_head),
        dtype=pe_in_dt
    )
    q_local = nl.ndarray(
        (q_seq_n_tiles, nl.par_dim(d_head_tile_size), q_seq_tile_size),
        dtype=pe_in_dt
    )
    k_local = nl.ndarray(
        (k_seq_n_tiles, nl.par_dim(d_head_tile_size), k_seq_tile_size),
        dtype=pe_in_dt
    )

    for i_seq_tile in nl.affine_range(v_seq_n_tiles):
        # V transpose
        trans_v[v_rows, i_seq_tile, head_cols] = nl.load(
            v_ref[i_seq_tile * v_seq_tile_size + v_rows, head_cols],
            dtype=pe_in_dt
        )
        # Q load+transpose+scale
        q_local[i_seq_tile, ip_q, if_q] = (
            nl.load_transpose2d(
                q_ref[i_seq_tile * q_seq_tile_size + seq_rows, head_cols],
                dtype=pe_in_dt
            ) * softmax_scale
        )
        # K load+transpose
        k_local[i_seq_tile, ip_k, if_k] = nl.load_transpose2d(
            k_ref[i_seq_tile * k_seq_tile_size + seq_rows, head_cols],
            dtype=pe_in_dt
        )

    # Main softmax+attention loop
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
        # QK‐buffer on SBUF in pe_in_dt
        qk_res_buf = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), seqlen),
            dtype=pe_in_dt,
            buffer=nl.sbuf
        )
        # If causal mask, prefill with large negative so exp→0
        if use_causal_mask:
            qk_res_buf[...] = -9984.0

        # Reduction outputs in kernel_dtype
        neg_max_res = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), k_seq_n_tiles),
            dtype=kernel_dtype
        )

        # Steps 2-4: compute Q·K^T, apply mask via predicated copy, partial row-max
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            base_col = i_k_seq_tile * k_seq_tile_size

            # 128×128 matmul into PSUM
            qk_psum = nl.zeros(
                (nl.par_dim(q_seq_tile_size), k_seq_tile_size),
                dtype=np.float32,
                buffer=nl.psum
            )
            qk_psum[ip_qk, if_qk] += nisa.nc_matmul(
                moving=k_local[i_k_seq_tile, ip_k, if_k],
                stationary=q_local[i_q_seq_tile, ip_q, if_q]
            )

            if use_causal_mask:
                # predicate True where global_q_row >= global_k_col
                pred_tile = (i_q_seq_tile * q_seq_tile_size + ip_qk
                             >= base_col + if_qk)
                # slice out the destination
                dst_slice = qk_res_buf[ip_qk, base_col + if_qk]
                # one pass: copy where pred True, leave -9984.0 otherwise
                nisa.tensor_copy_predicated(
                    src=qk_psum[ip_qk, if_qk],
                    dst=dst_slice,
                    predicate=pred_tile
                )
            else:
                # non-causal: copy entire PSUM tile
                qk_res_buf[ip_qk, base_col + if_qk] = \
                    nisa.tensor_copy(
                        qk_psum[ip_qk, if_qk],
                        dtype=pe_in_dt
                    )

            # Partial row max (negated)
            neg_max_res[ip_max, i_k_seq_tile] = nisa.tensor_reduce(
                np.max,
                data=qk_res_buf[ip_qk, base_col + if_qk],
                axis=(1,),
                dtype=kernel_dtype,
                negate=True
            )

        # Final row max
        neg_max_res_final = nisa.tensor_reduce(
            np.min,
            data=neg_max_res[ip_max, if_max],
            axis=(1,),
            dtype=kernel_dtype,
            negate=False
        )

        # Fused exp + sum
        sum_acc = nl.zeros(
            (nl.par_dim(q_seq_tile_size), 1),
            dtype=kernel_dtype,
            buffer=nl.sbuf
        )
        exp_res = nisa.activation(
            op=nl.exp,
            data=qk_res_buf[ip_softmax, if_softmax],
            bias=neg_max_res_final,
            scale=1.0,
            reduce_op=np.add,
            reduce_res=sum_acc,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
            dtype=pe_in_dt
        )

        # Compute normalization divisor
        sum_reciprocal = 1.0 / sum_acc

        # Transpose softmax results for V multiplication
        trans_softmax_res = nl.ndarray(
            (nl.par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
            dtype=pe_in_dt
        )
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            base_col = i_k_seq_tile * k_seq_tile_size
            trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = \
                nisa.nc_transpose(
                    exp_res[ip_scores, base_col + if_scores],
                    engine=nisa.tensor_engine
                )

        # Step 6: final QK-softmax @ V^T
        attn_res_psum = nl.zeros(
            (nl.par_dim(q_seq_tile_size), d_head_tile_size),
            dtype=np.float32,
            buffer=nl.psum
        )
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            attn_res_psum[ip_softmax, if_v_t] += nisa.nc_matmul(
                stationary=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                moving=trans_v[ip_v_t, i_k_seq_tile, if_v_t]
            )

        # Fuse PSUM→SBUF copy and per-Q scaling into one vector op
        attn_res_div = nisa.tensor_scalar(
            data=attn_res_psum[ip_softmax, if_v_t],
            op0=np.multiply,
            operand0=sum_reciprocal,   # broadcast full vector
            dtype=kernel_dtype
        )

        nl.store(
            out_ref[
                i_q_seq_tile * q_seq_tile_size + ip_softmax,
                if_v_t
            ],
            value=attn_res_div
        )

    return out_ref
