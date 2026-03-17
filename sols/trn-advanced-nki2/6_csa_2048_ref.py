@nki.jit
def test(q, k, v, causal_mask,
         kernel_dtype, acc_type,
         seq_len=2048,
         d_head=128):
    NEG_INF = -9984.0

    B_P_SIZE = 128   # queries per tile = d_head
    B_F_SIZE = 512   # keys per tile for QK
    REDUCTION_TILE = min(2048, seq_len // 2)
    num_k_tiles = seq_len // B_F_SIZE
    num_q_tiles = seq_len // B_P_SIZE

    o = nl.ndarray((seq_len, d_head), dtype=kernel_dtype, buffer=nl.shared_hbm)
    l = nl.ndarray((seq_len, 1), dtype=kernel_dtype, buffer=nl.shared_hbm)
    m = nl.ndarray((seq_len, 1), dtype=kernel_dtype, buffer=nl.shared_hbm)

    for q_tile_idx in nl.affine_range(num_q_tiles):
        q_start = q_tile_idx * B_P_SIZE
        q_end = q_start + B_P_SIZE

        # Load q tile: (B_P_SIZE, d_head)
        q_tile = nl.ndarray((B_P_SIZE, d_head), dtype=kernel_dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_tile, src=q[q_start:q_end, 0:d_head])

        # Transpose q_tile to (d_head, B_P_SIZE) for correct QK matmul
        q_tile_T_psum = nl.ndarray((d_head, B_P_SIZE), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=q_tile_T_psum, data=q_tile)
        q_tile_T = nl.ndarray((d_head, B_P_SIZE), dtype=kernel_dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_tile_T, src=q_tile_T_psum)

        qk_res_buf = nl.ndarray((B_P_SIZE, seq_len), buffer=nl.sbuf, dtype=acc_type)

        k_tile = nl.ndarray((d_head, B_F_SIZE), dtype=kernel_dtype, buffer=nl.sbuf)
        causal_tile = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)

        for k_i in nl.affine_range(num_k_tiles):
            k_start = k_i * B_F_SIZE
            k_end = k_start + B_F_SIZE

            # Causal check at tile level (Python-level, compile-time)
            right_of_diag = k_start > q_start + B_P_SIZE - 1  # all keys come after all queries

            if right_of_diag:
                # All values masked — write NEG_INF to this k-slice (write with dynamic offset works)
                nisa.memset(dst=qk_res_buf[0:B_P_SIZE, k_start:k_end], value=NEG_INF)
            else:
                # Compute QK
                nisa.dma_copy(dst=k_tile, src=k[0:d_head, k_start:k_end])
                qk_psum = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=qk_psum, stationary=q_tile_T, moving=k_tile)
                qk_sbuf = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=acc_type, buffer=nl.sbuf)
                nisa.tensor_copy(dst=qk_sbuf, src=qk_psum)

                left_of_diag = q_start >= k_end  # all queries come after all keys

                if left_of_diag:
                    qk_bf16 = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=kernel_dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=qk_bf16, src=qk_sbuf)
                    nisa.tensor_copy(dst=qk_res_buf[0:B_P_SIZE, k_start:k_end], src=qk_bf16)
                else:
                    # On the diagonal: apply causal mask
                    nisa.dma_copy(dst=causal_tile, src=causal_mask[q_start:q_end, k_start:k_end])
                    # masked = (qk - NEG_INF) * mask + NEG_INF
                    qk_shifted = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=acc_type, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=qk_shifted, data=qk_sbuf, op0=nl.add, operand0=-NEG_INF)
                    masked_shifted = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=acc_type, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=masked_shifted, data1=qk_shifted, data2=causal_tile, op=nl.multiply)
                    masked = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=acc_type, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=masked, data=masked_shifted, op0=nl.add, operand0=NEG_INF)
                    masked_bf16 = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=kernel_dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=masked_bf16, src=masked)
                    nisa.tensor_copy(dst=qk_res_buf[0:B_P_SIZE, k_start:k_end], src=masked_bf16)

        # Compute max over full qk_res_buf (no dynamic offset — reads from offset 0)
        max_ = nl.ndarray((B_P_SIZE, 1), dtype=acc_type, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=max_, op=nl.maximum, data=qk_res_buf, axis=1, keepdims=True)

        # Store m
        m_cast = nl.ndarray((B_P_SIZE, 1), dtype=kernel_dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=m_cast, src=max_)
        nisa.dma_copy(dst=m[q_start:q_end, 0:1], src=m_cast)

        # Compute p_local = exp(qk - max) in a single pass over full qk_res_buf (offset=0).
        # Using single-pass avoids the NKI issue where nisa.activation reads from
        # dynamic-offset SBUF slices always use offset=0.
        neg_max = nl.ndarray((B_P_SIZE, 1), dtype=acc_type, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=neg_max, data=max_, op0=nl.multiply, operand0=-1.0)

        p_local = nl.ndarray((B_P_SIZE, seq_len), dtype=kernel_dtype, buffer=nl.sbuf)
        nisa.activation(
            dst=p_local,
            op=nl.exp,
            data=qk_res_buf,
            bias=neg_max, scale=1.0,
        )

        ps = nl.ndarray((B_P_SIZE, 1), dtype=acc_type, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ps, op=nl.add, data=p_local, axis=1, keepdims=True)

        # Compute PV: for each 128-key block, transpose p_slice then matmul with v
        pv_accum = nl.ndarray((B_P_SIZE, d_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=pv_accum, value=0.0)

        p_slice_T = nl.ndarray((B_P_SIZE, B_P_SIZE), dtype=kernel_dtype, buffer=nl.sbuf)
        v_tile = nl.ndarray((B_P_SIZE, d_head), dtype=kernel_dtype, buffer=nl.sbuf)

        for k_i in nl.affine_range(seq_len // B_P_SIZE):
            k_start = k_i * B_P_SIZE
            # Transpose p_slice
            p_slice_T_psum = nl.ndarray((B_P_SIZE, B_P_SIZE), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=p_slice_T_psum,
                             data=p_local[0:B_P_SIZE, k_start:k_start+B_P_SIZE])
            nisa.tensor_copy(dst=p_slice_T, src=p_slice_T_psum)

            # Load v tile: (B_P_SIZE, d_head)
            nisa.dma_copy(dst=v_tile, src=v[k_start:k_start+B_P_SIZE, 0:d_head])

            # PV contribution
            pv_psum = nl.ndarray((B_P_SIZE, d_head), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=pv_psum, stationary=p_slice_T, moving=v_tile)
            tmp = nl.ndarray((B_P_SIZE, d_head), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=tmp, src=pv_psum)
            nisa.tensor_tensor(dst=pv_accum, data1=pv_accum, data2=tmp, op=nl.add)

        # Store o
        o_cast = nl.ndarray((B_P_SIZE, d_head), dtype=kernel_dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_cast, src=pv_accum)
        nisa.dma_copy(dst=o[q_start:q_end, 0:d_head], src=o_cast)

        # Store l = log(ps) + max
        log_ps = nl.ndarray((B_P_SIZE, 1), dtype=acc_type, buffer=nl.sbuf)
        nisa.activation(dst=log_ps, op=nl.log, data=ps)
        l_val = nl.ndarray((B_P_SIZE, 1), dtype=acc_type, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=l_val, data1=log_ps, data2=max_, op=nl.add)
        l_cast = nl.ndarray((B_P_SIZE, 1), dtype=kernel_dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=l_cast, src=l_val)
        nisa.dma_copy(dst=l[q_start:q_end, 0:1], src=l_cast)

    return o, l, m
