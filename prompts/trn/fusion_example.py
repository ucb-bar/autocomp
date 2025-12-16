def PROMPT():
    return """Here is an example of a fused kernel that inlines two matrix multiplications into a single loop to enable SBUF residency (among other optimizations).
@nki.jit
def nki_fused_all_proj_kernel(x_tensor, g_tensor, up_wT, gate_wT, down_wT):
    '''
    Fused RMSNorm + up/gate projections + SiLU(gate)*up + down projection.

    Inputs:
      x_tensor : [R, H]
      g_tensor : [H]
      up_wT    : [H, U]   (transposed weight; K-major)
      gate_wT  : [H, U]   (transposed weight; K-major)
      down_wT  : [U, D]   (transposed weight; K-major)

    Output:
      out      : [R, D]
    '''
    R, H = x_tensor.shape
    H2, U = up_wT.shape
    H3, U2 = gate_wT.shape
    U3, D = down_wT.shape
    assert H2 == H and H3 == H and U2 == U and U3 == U

    out = nl.ndarray((R, D), dtype=x_tensor.dtype, buffer=nl.shared_hbm)

    # Indices used for loads
    i_h = nl.arange(H)[None, :]          # [1, H]
    i_one = nl.arange(1)[:, None]        # [1, 1]

    # Load RMSNorm scale once into SBUF as [1, H]
    g_tile = nl.load(g_tensor.reshape((1, H))[i_one, i_h])  # [1, H]

    # Row tiling
    P_TILE = min(nl.tile_size.pmax, R)   # compile-time constant per specialization
    i_p = nl.arange(P_TILE)[:, None]     # [P_TILE, 1]
    trip = (R + P_TILE - 1) // P_TILE

    # Common tile sizes
    TILE_K = nl.tile_size.pmax  # 128 (contraction for H and for down\'s K-subtiles)

    # Down-proj tiling (assume common LLM shape: D multiple of 512)
    TILE_D = nl.tile_size.gemm_moving_fmax  # typically 512
    assert (D % TILE_D) == 0
    num_d = D // TILE_D
    # Keep on-chip accumulator count bounded (covers D up to 2048 with TILE_D=512)
    assert num_d <= 4

    # Up/gate output tiling
    do_swap = (P_TILE < nl.tile_size.pmax) and (U >= nl.tile_size.pmax)

    # H tiling (assume common LLM shape: H multiple of 128)
    assert (H % TILE_K) == 0
    num_k = H // TILE_K

    for p in nl.affine_range(trip):
        row_idx = p * P_TILE + i_p                # [P_TILE, 1]
        row_mask = (row_idx < R)                  # [P_TILE, 1]

        # Load one [P_TILE, H] tile
        x_tile = nl.load(x_tensor[row_idx, i_h], mask=row_mask)

        # RMSNorm: y = x / rms(x) * g
        sq = nl.square(x_tile, mask=row_mask)
        sq_sum = nl.sum(sq, axis=[1], mask=row_mask)            # [P_TILE, 1]
        mean = sq_sum / float(H)                                # [P_TILE, 1]
        inv_rms = nl.rsqrt(mean + 1.0e-5, mask=row_mask)         # [P_TILE, 1]
        y_tile = nl.multiply(x_tile, inv_rms, mask=row_mask)
        y_tile = nl.multiply(y_tile, g_tile, mask=row_mask)      # g_tile broadcasts on partition

        # Down-proj accumulators (keep in PSUM across all U tiles)
        acc_down0 = nl.zeros((nl.par_dim(P_TILE), TILE_D), dtype=nl.float32, buffer=nl.psum)
        acc_down1 = nl.zeros((nl.par_dim(P_TILE), TILE_D), dtype=nl.float32, buffer=nl.psum)
        acc_down2 = nl.zeros((nl.par_dim(P_TILE), TILE_D), dtype=nl.float32, buffer=nl.psum)
        acc_down3 = nl.zeros((nl.par_dim(P_TILE), TILE_D), dtype=nl.float32, buffer=nl.psum)

        if do_swap:
            # Swap path for skinny M: produce [M, 128] act tiles
            TILE_U = nl.tile_size.pmax  # 128
            assert (U % TILE_U) == 0
            num_n = U // TILE_U

            # Precompute y^T blocks in SBUF:
            # yT_cat: [Ktile=128, num_k * M], slice out [128, M] per k
            yT_cat = nl.ndarray((nl.par_dim(TILE_K), num_k * P_TILE),
                                dtype=x_tensor.dtype, buffer=nl.sbuf)

            for k in nl.affine_range(num_k):
                k0 = k * TILE_K
                y_block = y_tile[:, nl.ds(k0, TILE_K)]  # [M, 128]
                yT_psum = nisa.nc_transpose(y_block, engine=nisa.tensor_engine)  # PSUM [128, M]
                yT_sb = nisa.tensor_copy(yT_psum, dtype=x_tensor.dtype)          # SBUF [128, M]
                yT_cat[:, nl.ds(k * P_TILE, P_TILE)] = yT_sb

            for n in nl.affine_range(num_n):
                n0 = n * TILE_U

                # up/gate accumulators in swapped layout: [128, M] on PSUM
                acc_up = nl.zeros((nl.par_dim(TILE_U), P_TILE), dtype=nl.float32, buffer=nl.psum)
                acc_gate = nl.zeros((nl.par_dim(TILE_U), P_TILE), dtype=nl.float32, buffer=nl.psum)

                for k in nl.affine_range(num_k):
                    k0 = k * TILE_K
                    lhsT = yT_cat[:, nl.ds(k * P_TILE, P_TILE)]  # [K=128, M]
                    up_w_tile = nl.load(up_wT[nl.ds(k0, TILE_K), nl.ds(n0, TILE_U)])        # [K=128, N=128]
                    gate_w_tile = nl.load(gate_wT[nl.ds(k0, TILE_K), nl.ds(n0, TILE_U)])    # [K=128, N=128]
                    acc_up += nl.matmul(up_w_tile, lhsT, transpose_x=True)                  # [N, M]
                    acc_gate += nl.matmul(gate_w_tile, lhsT, transpose_x=True)              # [N, M]

                # Convert to [M, 128] in SBUF for SiLU/multiply
                up_nm_sb = nisa.tensor_copy(acc_up, dtype=x_tensor.dtype)       # SBUF [128, M]
                gate_nm_sb = nisa.tensor_copy(acc_gate, dtype=x_tensor.dtype)   # SBUF [128, M]

                up_mn_psum = nisa.nc_transpose(up_nm_sb, engine=nisa.tensor_engine)         # PSUM [M, 128]
                gate_mn_psum = nisa.nc_transpose(gate_nm_sb, engine=nisa.tensor_engine)     # PSUM [M, 128]
                up_mn = nisa.tensor_copy(up_mn_psum, dtype=x_tensor.dtype)                  # SBUF [M, 128]
                gate_mn = nisa.tensor_copy(gate_mn_psum, dtype=x_tensor.dtype)              # SBUF [M, 128]

                gate_silu = nisa.activation(op=nl.silu, data=gate_mn, dtype=gate_mn.dtype)  # SBUF [M, 128]
                act_mn = nl.multiply(up_mn, gate_silu)                                      # SBUF [M, 128]

                # Down-proj update: act_mn [M,128] @ down_wT[n0:n0+128, d0:d0+512]
                down0 = nl.load(down_wT[nl.ds(n0, TILE_U), nl.ds(0 * TILE_D, TILE_D)])
                acc_down0 += nl.matmul(act_mn, down0)

                if num_d > 1:
                    down1 = nl.load(down_wT[nl.ds(n0, TILE_U), nl.ds(1 * TILE_D, TILE_D)])
                    acc_down1 += nl.matmul(act_mn, down1)
                if num_d > 2:
                    down2 = nl.load(down_wT[nl.ds(n0, TILE_U), nl.ds(2 * TILE_D, TILE_D)])
                    acc_down2 += nl.matmul(act_mn, down2)
                if num_d > 3:
                    down3 = nl.load(down_wT[nl.ds(n0, TILE_U), nl.ds(3 * TILE_D, TILE_D)])
                    acc_down3 += nl.matmul(act_mn, down3)

        else:
            # Standard path: compute up/gate in [M, 512] tiles, then split into 128-K chunks for down
            TILE_U = nl.tile_size.gemm_moving_fmax  # typically 512
            assert (U % TILE_U) == 0
            num_n = U // TILE_U

            for n in nl.affine_range(num_n):
                n0 = n * TILE_U

                acc_up = nl.zeros((nl.par_dim(P_TILE), TILE_U), dtype=nl.float32, buffer=nl.psum)
                acc_gate = nl.zeros((nl.par_dim(P_TILE), TILE_U), dtype=nl.float32, buffer=nl.psum)

                for k in nl.affine_range(num_k):
                    k0 = k * TILE_K
                    x_block = y_tile[:, nl.ds(k0, TILE_K)]                                     # [M, 128]
                    up_w_tile = nl.load(up_wT[nl.ds(k0, TILE_K), nl.ds(n0, TILE_U)])           # [128, 512]
                    gate_w_tile = nl.load(gate_wT[nl.ds(k0, TILE_K), nl.ds(n0, TILE_U)])       # [128, 512]
                    acc_up += nl.matmul(x_block, up_w_tile)                                    # [M, 512]
                    acc_gate += nl.matmul(x_block, gate_w_tile)                                # [M, 512]

                up_sb = nisa.tensor_copy(acc_up, dtype=x_tensor.dtype)                          # SBUF [M, 512]
                gate_sb = nisa.tensor_copy(acc_gate, dtype=x_tensor.dtype)                      # SBUF [M, 512]
                gate_silu = nisa.activation(op=nl.silu, data=gate_sb, dtype=gate_sb.dtype)      # SBUF [M, 512]
                act_sb = nl.multiply(up_sb, gate_silu)                                          # SBUF [M, 512]

                # Down-proj in 4x128 chunks to satisfy matmul K<=128
                # act_blk: [M,128], down_blk: [128,512] => [M,512]
                for u_sub in range(0, TILE_U, TILE_K):
                    u_base = n0 + u_sub
                    act_blk = act_sb[:, nl.ds(u_sub, TILE_K)]                                   # [M, 128]

                    down0 = nl.load(down_wT[nl.ds(u_base, TILE_K), nl.ds(0 * TILE_D, TILE_D)])
                    acc_down0 += nl.matmul(act_blk, down0)

                    if num_d > 1:
                        down1 = nl.load(down_wT[nl.ds(u_base, TILE_K), nl.ds(1 * TILE_D, TILE_D)])
                        acc_down1 += nl.matmul(act_blk, down1)
                    if num_d > 2:
                        down2 = nl.load(down_wT[nl.ds(u_base, TILE_K), nl.ds(2 * TILE_D, TILE_D)])
                        acc_down2 += nl.matmul(act_blk, down2)
                    if num_d > 3:
                        down3 = nl.load(down_wT[nl.ds(u_base, TILE_K), nl.ds(3 * TILE_D, TILE_D)])
                        acc_down3 += nl.matmul(act_blk, down3)

        # Store down-proj output tiles
        i_d = nl.arange(TILE_D)[None, :]

        out0_sb = nisa.tensor_copy(acc_down0, dtype=x_tensor.dtype)
        nl.store(out[row_idx, 0 * TILE_D + i_d], value=out0_sb, mask=row_mask)

        if num_d > 1:
            out1_sb = nisa.tensor_copy(acc_down1, dtype=x_tensor.dtype)
            nl.store(out[row_idx, 1 * TILE_D + i_d], value=out1_sb, mask=row_mask)
        if num_d > 2:
            out2_sb = nisa.tensor_copy(acc_down2, dtype=x_tensor.dtype)
            nl.store(out[row_idx, 2 * TILE_D + i_d], value=out2_sb, mask=row_mask)
        if num_d > 3:
            out3_sb = nisa.tensor_copy(acc_down3, dtype=x_tensor.dtype)
            nl.store(out[row_idx, 3 * TILE_D + i_d], value=out3_sb, mask=row_mask)

    return out


def test(x, post_attention_layernorm_weight, up_proj_weight, gate_proj_weight, down_proj_weight):
    # Shapes (prefill example):
    # x: (1, 32, 2048)
    # weights: up/gate (8192, 2048), down (2048, 8192)
    b, s, h = x.shape

    # Flatten batch*seq to match kernels (R, H)
    x2d = x.view(-1, h)

    # One kernel: RMSNorm + up/gate + SiLU*up + down
    output = nki_fused_all_proj_kernel(
        x2d,
        post_attention_layernorm_weight,
        up_proj_weight.t(),
        gate_proj_weight.t(),
        down_proj_weight.t(),
    )  # (R, 2048)

    return output
"""