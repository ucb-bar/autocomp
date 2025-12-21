@nki.jit
def test(x_tensor, gamma, ug_wT, down_wT):
    """
    Inputs:
      x_tensor : [R, H]
      gamma    : [H]
      ug_wT    : [H, 2*U] packed weights (Up+Gate), block-interleaved by PACK_TILE_U:
                 for each n0 in [0, U) step PACK_TILE_U:
                   ug_wT[:, 2*n0 : 2*n0+PACK_TILE_U]           = up_wT[:, n0:n0+PACK_TILE_U]
                   ug_wT[:, 2*n0+PACK_TILE_U : 2*n0+2*PACK_TILE_U] = gate_wT[:, n0:n0+PACK_TILE_U]
      down_wT  : [U, D]   (transposed weight; K-major for matmul)
    Output:
      out      : [R, D]
    """
    R, H = x_tensor.shape
    H2, ug_cols = ug_wT.shape
    U3, D = down_wT.shape

    assert H2 == H
    assert (ug_cols % 2) == 0
    U = ug_cols // 2
    assert U3 == U

    out = nl.ndarray((R, D), dtype=x_tensor.dtype, buffer=nl.shared_hbm)

    TILE_K = nl.tile_size.pmax                 # 128
    TILE_D = nl.tile_size.gemm_moving_fmax     # typically 512
    assert (H % TILE_K) == 0
    assert (U % TILE_K) == 0
    assert (D % TILE_D) == 0

    num_k = H // TILE_K
    num_d = D // TILE_D

    # Prefer packed blocks of 256 (gives fused RHS free dim=512); fallback to 128 if needed.
    PACK_TILE_U = 256 if (U % 256) == 0 else 128
    assert (U % PACK_TILE_U) == 0

    # Row tiling
    P_TILE = min(nl.tile_size.pmax, R)         # <= 128, specialization-time constant
    i_p = nl.arange(P_TILE)[:, None]           # [P_TILE, 1]
    i_h = nl.arange(H)[None, :]                # [1, H]
    i_d = nl.arange(TILE_D)[None, :]           # [1, TILE_D]

    # Load gamma once into SBUF as [1, H] for broadcast multiply
    g_tile = nl.load(gamma.reshape((1, H))[0:1, nl.ds(0, H)])  # [1, H]

    # Swap path whenever M (rows) is not full 128
    do_swap = (P_TILE < nl.tile_size.pmax)

    if do_swap:
        # --- Small-M path (Optimized: fused K-loop) ---
        trip = (R + P_TILE - 1) // P_TILE

        # Drive n0 in a way that matches packing blocks without requiring div/mod on affine indices.
        pack_blocks = U // PACK_TILE_U
        halves_per_pack = PACK_TILE_U // TILE_K  # 1 if PACK_TILE_U=128, 2 if 256

        for p in nl.affine_range(trip):
            row_idx = p * P_TILE + i_p
            row_mask = (row_idx < R)

            # Load [P_TILE, H]
            x_tile = nl.load(x_tensor[row_idx, i_h], mask=row_mask)

            # RMSNorm
            sq = nl.square(x_tile, mask=row_mask)
            sq_sum = nl.sum(sq, axis=[1], mask=row_mask)          # [P_TILE, 1]
            mean = sq_sum / float(H)
            inv_rms = nl.rsqrt(mean + 1.0e-5, mask=row_mask)      # [P_TILE, 1]
            y_tile = nl.multiply(x_tile, inv_rms, mask=row_mask)  # [P_TILE, H]
            y_tile = nl.multiply(y_tile, g_tile, mask=row_mask)   # [P_TILE, H]

            # Output accumulator (float32) on SBUF: [P_TILE, D]
            # Allocated per p-loop to start fresh with zeros
            out_acc = nl.zeros((nl.par_dim(P_TILE), D), dtype=nl.float32, buffer=nl.sbuf)

            # Compute projections in 128-wide chunks.
            # We fuse the K-loop inside the U-block loop to reuse the transposed Y block
            # and avoid storing a large yT_cat tensor in SBUF.
            for nb in nl.affine_range(pack_blocks):
                pack_u0 = nb * PACK_TILE_U
                ug_col0 = nb * (2 * PACK_TILE_U)
                
                for hlf in range(halves_per_pack):
                    n0 = pack_u0 + hlf * TILE_K
                    up_col0 = ug_col0 + hlf * TILE_K
                    gt_col0 = ug_col0 + PACK_TILE_U + hlf * TILE_K

                    # Allocate per-block accumulators in PSUM
                    acc_up = nl.zeros((nl.par_dim(TILE_K), P_TILE), dtype=nl.float32, buffer=nl.psum)
                    acc_gate = nl.zeros((nl.par_dim(TILE_K), P_TILE), dtype=nl.float32, buffer=nl.psum)

                    # FUSED single pass over K
                    for k in nl.affine_range(num_k):
                        k0 = k * TILE_K
                        
                        # 1) Load the P_TILE x TILE_K slab of Y
                        y_blk = y_tile[:, nl.ds(k0, TILE_K)]  # [P_TILE, 128]

                        # 2) Transpose it PF->FP on TensorEngine and copy back to SBUF
                        # Input P_TILE <= 128, Tensor Engine supports shape <= 128x128
                        yT_psum = nisa.nc_transpose(y_blk, engine=nisa.tensor_engine)  # [128, P_TILE] on PSUM
                        yT_sb = nisa.tensor_copy(yT_psum, dtype=x_tensor.dtype)        # [128, P_TILE] on SBUF

                        # 3) Load corresponding weight blocks
                        up_w = nl.load(ug_wT[nl.ds(k0, TILE_K), nl.ds(up_col0, TILE_K)])
                        gate_w = nl.load(ug_wT[nl.ds(k0, TILE_K), nl.ds(gt_col0, TILE_K)])

                        # 4) Accumulate projections
                        acc_up += nl.matmul(up_w, yT_sb, transpose_x=True)
                        acc_gate += nl.matmul(gate_w, yT_sb, transpose_x=True)

                    # 5) Transpose back to [P_TILE, 128], do silu gating
                    up_nm_sb = nisa.tensor_copy(acc_up, dtype=x_tensor.dtype)
                    gate_nm_sb = nisa.tensor_copy(acc_gate, dtype=x_tensor.dtype)

                    up_mn = nisa.tensor_copy(nisa.nc_transpose(up_nm_sb, engine=nisa.tensor_engine), dtype=x_tensor.dtype)
                    gate_mn = nisa.tensor_copy(nisa.nc_transpose(gate_nm_sb, engine=nisa.tensor_engine), dtype=x_tensor.dtype)

                    gate_silu = nisa.activation(op=nl.silu, data=gate_mn, dtype=gate_mn.dtype)
                    act_mn = nl.multiply(up_mn, gate_silu)  # [P_TILE, 128]

                    # 6) Down-projection into out_acc
                    for di in nl.affine_range(num_d):
                        d0 = di * TILE_D
                        down_blk = nl.load(down_wT[nl.ds(n0, TILE_K), nl.ds(d0, TILE_D)])  # [128, 512]
                        res = nl.matmul(act_mn, down_blk)  # [P_TILE, 512]
                        out_acc[:, nl.ds(d0, TILE_D)] += res

            # Store result
            for di in nl.affine_range(num_d):
                d0 = di * TILE_D
                out_tile = nisa.tensor_copy(out_acc[:, nl.ds(d0, TILE_D)], dtype=x_tensor.dtype)
                nl.store(out[row_idx, d0 + i_d], value=out_tile, mask=row_mask)

    else:
        # --- Standard path (P_TILE == 128): fused Up+Gate projection GEMM using packed ug_wT ---
        # Force TILE_U=256 when possible to make FUSED_N=512 (Tensor Engine max RHS free dim).
        TILE_U = 256 if (U % 256) == 0 else 128
        FUSED_N = 2 * TILE_U
        assert (U % TILE_U) == 0
        assert FUSED_N <= nl.tile_size.gemm_moving_fmax  # 512

        num_us = TILE_U // TILE_K
        num_n = U // TILE_U

        P_BLOCK = 2
        trip = (R + P_TILE - 1) // P_TILE
        trip_pblk = (trip + P_BLOCK - 1) // P_BLOCK

        for pb in nl.affine_range(trip_pblk):
            p0 = pb * 2
            p1 = pb * 2 + 1

            row_idx0 = p0 * P_TILE + i_p
            row_idx1 = p1 * P_TILE + i_p
            row_mask0 = (row_idx0 < R)
            row_mask1 = (row_idx1 < R)

            # RMSNorm -> y0, y1
            x0 = nl.load(x_tensor[row_idx0, i_h], mask=row_mask0)
            x1 = nl.load(x_tensor[row_idx1, i_h], mask=row_mask1)

            sq0 = nl.square(x0, mask=row_mask0)
            sq1 = nl.square(x1, mask=row_mask1)

            ss0 = nl.sum(sq0, axis=[1], mask=row_mask0)  # [128, 1]
            ss1 = nl.sum(sq1, axis=[1], mask=row_mask1)  # [128, 1]

            inv0 = nl.rsqrt((ss0 / float(H)) + 1.0e-5, mask=row_mask0)
            inv1 = nl.rsqrt((ss1 / float(H)) + 1.0e-5, mask=row_mask1)

            y0 = nl.multiply(nl.multiply(x0, inv0, mask=row_mask0), g_tile, mask=row_mask0)
            y1 = nl.multiply(nl.multiply(x1, inv1, mask=row_mask1), g_tile, mask=row_mask1)

            # Output accumulators (float32) in SBUF: [128, D]
            out0 = nl.zeros((nl.par_dim(P_TILE), D), dtype=nl.float32, buffer=nl.sbuf)
            out1 = nl.zeros((nl.par_dim(P_TILE), D), dtype=nl.float32, buffer=nl.sbuf)

            # Double-buffered fused weight cache:
            # Shape per buffer: [K=128, num_k*FUSED_N]
            ug_cache = nl.ndarray((2, nl.par_dim(TILE_K), num_k * FUSED_N),
                                  dtype=x_tensor.dtype, buffer=nl.sbuf)

            # Prologue: load weights for n=0 into buffer 0
            if num_n > 0:
                n0 = 0
                ug_col0 = 2 * n0  # packed column base
                for k in nl.affine_range(num_k):
                    k0 = k * TILE_K
                    ug_cache[0, :, nl.ds(k * FUSED_N, FUSED_N)] = nl.load(
                        ug_wT[nl.ds(k0, TILE_K), nl.ds(ug_col0, FUSED_N)]
                    )

            # Pipelined sequential loop over n (due to double-buffer prefetch dependency)
            for n in nl.sequential_range(num_n):
                curr = n % 2
                nxt = 1 - curr

                # Prefetch weights for n+1 into nxt buffer (overlaps with compute below)
                if n < (num_n - 1):
                    n1 = (n + 1) * TILE_U
                    ug_col1 = 2 * n1
                    for k in nl.affine_range(num_k):
                        k0 = k * TILE_K
                        ug_cache[nxt, :, nl.ds(k * FUSED_N, FUSED_N)] = nl.load(
                            ug_wT[nl.ds(k0, TILE_K), nl.ds(ug_col1, FUSED_N)]
                        )

                # Fused projection accumulators: [128, FUSED_N] on PSUM
                acc_ug0 = nl.zeros((nl.par_dim(P_TILE), FUSED_N), dtype=nl.float32, buffer=nl.psum)
                acc_ug1 = nl.zeros((nl.par_dim(P_TILE), FUSED_N), dtype=nl.float32, buffer=nl.psum)

                for k in nl.affine_range(num_k):
                    k0 = k * TILE_K
                    xblk0 = y0[:, nl.ds(k0, TILE_K)]  # [128, 128]
                    xblk1 = y1[:, nl.ds(k0, TILE_K)]  # [128, 128]

                    ug_w = ug_cache[curr, :, nl.ds(k * FUSED_N, FUSED_N)]  # [128, FUSED_N]
                    acc_ug0 += nl.matmul(xblk0, ug_w)  # [128, FUSED_N]
                    acc_ug1 += nl.matmul(xblk1, ug_w)  # [128, FUSED_N]

                ug0 = nisa.tensor_copy(acc_ug0, dtype=x_tensor.dtype)  # [128, FUSED_N] SBUF
                ug1 = nisa.tensor_copy(acc_ug1, dtype=x_tensor.dtype)

                up0 = ug0[:, nl.ds(0, TILE_U)]
                gt0 = ug0[:, nl.ds(TILE_U, TILE_U)]
                up1 = ug1[:, nl.ds(0, TILE_U)]
                gt1 = ug1[:, nl.ds(TILE_U, TILE_U)]

                act0 = nl.multiply(up0, nisa.activation(op=nl.silu, data=gt0, dtype=gt0.dtype))
                act1 = nl.multiply(up1, nisa.activation(op=nl.silu, data=gt1, dtype=gt1.dtype))

                # Down projection (stream down_wT from HBM; reuse each down tile for both p0 and p1)
                n_base = n * TILE_U
                for us in nl.affine_range(num_us):
                    u_base = n_base + us * TILE_K
                    a0 = act0[:, nl.ds(us * TILE_K, TILE_K)]  # [128, 128]
                    a1 = act1[:, nl.ds(us * TILE_K, TILE_K)]  # [128, 128]

                    for di in nl.affine_range(num_d):
                        d0 = di * TILE_D
                        down_tile = nl.load(down_wT[nl.ds(u_base, TILE_K), nl.ds(d0, TILE_D)])  # [128, 512]
                        r0 = nisa.nc_matmul(a0, down_tile)  # [128, 512] on PSUM
                        r1 = nisa.nc_matmul(a1, down_tile)  # [128, 512] on PSUM
                        out0[:, nl.ds(d0, TILE_D)] += r0
                        out1[:, nl.ds(d0, TILE_D)] += r1

            # Store p0 / p1 results
            for di in nl.affine_range(num_d):
                d0 = di * TILE_D
                o0 = nisa.tensor_copy(out0[:, nl.ds(d0, TILE_D)], dtype=x_tensor.dtype)
                o1 = nisa.tensor_copy(out1[:, nl.ds(d0, TILE_D)], dtype=x_tensor.dtype)
                nl.store(out[row_idx0, d0 + i_d], value=o0, mask=row_mask0)
                nl.store(out[row_idx1, d0 + i_d], value=o1, mask=row_mask1)

    return out
