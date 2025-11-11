@nki.jit
def test(q, k, v):
    # q, k, v : [P=d_head=128, F=seqlen_q]
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax                       # 128
    FMAX_S = nl.tile_size.gemm_stationary_fmax     # <= 128
    FMAX_M = nl.tile_size.gemm_moving_fmax         # <= 512

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= FMAX_S

    # Output in HBM
    kernel_out = nl.ndarray((seqlen_q, d_head),
                            dtype=q.dtype,
                            buffer=nl.shared_hbm)

    # 0) Load full Q/K once into SBUF
    q_sbuf = nl.load(q)  # [128, seqlen_q]
    k_sbuf = nl.load(k)  # [128, seqlen_kv]

    # Precompute V^T sub‐tiles into SBUF to avoid repeated PF-transpose of V later
    num_kv_blocks       = seqlen_kv // FMAX_M
    kv_subtiles_per_blk = FMAX_M // PMAX            # typically 4
    total_kv128         = num_kv_blocks * kv_subtiles_per_blk  # = seqlen_kv // 128

    # shape: [total_kv128, P=128, free=d_head]
    vT_sbuf = nl.ndarray((total_kv128,
                          nl.par_dim(PMAX),
                          d_head),
                         dtype=nl.bfloat16,
                         buffer=nl.sbuf)
    for col in nl.affine_range(total_kv128):
        v_sub = nl.load(v[:, nl.ds(col * PMAX, PMAX)])        # [128,128]
        v_sub_T = nisa.nc_transpose(v_sub, dtype=nl.bfloat16) # [128,128] PSUM→SBUF
        vT_sbuf[col, :, :] = v_sub_T

    # Number of query tiles (assumes FMAX_S is chosen as 128 in practice)
    n_q_tiles = seqlen_q // PMAX

    # --- PASS A (fused block‐max) ---
    # Initialize running per-row max to -∞
    NEG_INF = float("-1e38")
    row_max_cur = nl.full((nl.par_dim(PMAX), 1),
                          fill_value=NEG_INF,
                          dtype=nl.float32,
                          buffer=nl.sbuf)

    # Sequential reduction over kv-blocks to update row_max_cur
    for ik in nl.sequential_range(num_kv_blocks):
        k_tile = k_sbuf[0:PMAX, nl.ds(ik * FMAX_M, FMAX_M)]  # [128,512]
        qk_psum = nisa.nc_matmul(
            stationary=q_sbuf[0:PMAX, nl.ds(0, FMAX_S)],
            moving=k_tile,
            is_transpose=True
        )  # [128,512] PSUM

        blk_max = nisa.tensor_reduce(
            op=nl.maximum,
            data=qk_psum,
            axis=[1],
            keepdims=True,
            dtype=nl.float32
        )  # [128,1] SBUF

        # fused update: row_max_cur = max(row_max_cur, blk_max)
        row_max_cur[:, :] = nl.maximum(row_max_cur, blk_max)

    # --- PASS B & C: recompute QK, form numerator & denominator, and accumulate attention ---
    for iq in nl.affine_range(n_q_tiles):
        # Allocate per-query buffers
        attn_out = nl.zeros((PMAX, d_head),
                            dtype=nl.float32,
                            buffer=nl.psum)
        sum_kv128 = nl.zeros((nl.par_dim(PMAX),
                              total_kv128),
                             dtype=nl.float32,
                             buffer=nl.sbuf)

        q_tile = q_sbuf[0:PMAX, nl.ds(iq * FMAX_S, FMAX_S)]

        for ik in nl.affine_range(num_kv_blocks):
            k_tile = k_sbuf[0:PMAX, nl.ds(ik * FMAX_M, FMAX_M)]
            qk_psum = nisa.nc_matmul(
                stationary=q_tile,
                moving=k_tile,
                is_transpose=True
            )  # [128,512] PSUM

            for j in nl.affine_range(kv_subtiles_per_blk):
                col = ik * kv_subtiles_per_blk + j

                qk_sub_psum = qk_psum[0:PMAX, nl.ds(j * PMAX, PMAX)]
                exp_sub = nisa.activation(
                    op=nl.exp,
                    data=qk_sub_psum,
                    bias=nl.negative(row_max_cur),  # [128,1]
                    dtype=nl.bfloat16
                )  # [128,128] SBUF

                sum_kv128[:, nl.ds(col, 1)] = nisa.tensor_reduce(
                    op=nl.add,
                    data=exp_sub,
                    axis=[1],
                    keepdims=True,
                    dtype=nl.float32
                )  # [128,1] SBUF

                exp_sub_T = nisa.nc_transpose(
                    exp_sub,
                    dtype=nl.bfloat16
                )  # [128,128] PSUM

                v_sub_T = vT_sbuf[col, :, :]  # [128, d_head] SBUF
                attn_out += nisa.nc_matmul(
                    stationary=exp_sub_T,
                    moving=v_sub_T
                )

        # final denom per‐row sum => [128,1]
        sum_row_cur = nisa.tensor_reduce(
            op=nl.add,
            data=sum_kv128,
            axis=[1],
            keepdims=True,
            dtype=nl.float32
        )

        inv_sum = nl.reciprocal(sum_row_cur)  # [128,1]

        # scale attn_out by inv_sum
        attn_out[:, :] = nisa.tensor_scalar(
            data=attn_out,
            op0=nl.multiply,
            operand0=inv_sum,
            engine=nisa.vector_engine,
            dtype=nl.float32
        )

        # write back to HBM
        nl.store(
            kernel_out[nl.ds(iq * PMAX, PMAX), :],
            attn_out
        )

    return kernel_out
