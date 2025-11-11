@nki.jit
def test(img_ref, filter_ref, **kwargs):
    # --- inline helper -------------------------------------------------------
    def _div_ceil(n, d):
        return (n + d - 1) // d

    # --- kernel setup --------------------------------------------------------
    padding = kwargs['padding']
    W_padding_l, W_padding_r = padding[1]

    N, C_in, H, W = img_ref.shape
    C_out, _, H_f, W_f = filter_ref.shape

    K0 = H - H_f + 1
    K1 = W + W_padding_l + W_padding_r - W_f + 1
    out_image_size = K0 * K1
    image_size = H * (W + W_padding_l + W_padding_r)
    dtype = img_ref.dtype

    # allocate final output in HBM
    out_hbm = nl.ndarray((N, C_out, K0, K1),
                         dtype=dtype,
                         buffer=nl.shared_hbm)

    # channel tiling parameters (same as original)
    C_NUM_TILES = _div_ceil(C_in, 128)
    C_TILE_SIZE = min(C_in, 128)

    # common index tiles
    i_p      = nl.arange(C_TILE_SIZE)[:, None]  # (C_TILE_SIZE, 1), partition axis
    i_f_k1   = nl.arange(K1)[None, :]           # (1, K1)
    i_f_a    = nl.arange(W_f)[None, :]          # (1, W_f)
    window_size = H_f * W_f
    i_f_win  = nl.arange(window_size)[None, :]  # (1, window_size)

    # --- tile over input channels --------------------------------------------
    for c_tile in nl.affine_range(C_NUM_TILES):
        base_chan = c_tile * C_TILE_SIZE

        # --- Phase A: load + copy one filter tile into PSUM ------------------
        # SBUF staging for this c_tile
        filt_sbuf = nl.zeros((nl.par_dim(C_TILE_SIZE), window_size),
                             dtype=dtype,
                             buffer=nl.sbuf)
        # flatten filter along spatial dimension (only H_f==1 row as original)
        i_wf = nl.arange(W_f)[None, :]
        filt_sbuf[i_p, i_wf * H_f + 0] = nl.load(
            filter_ref[i_p + base_chan, 0, 0, i_wf]
        )
        # copy into PSUM for reuse
        filt_psum = nl.zeros((nl.par_dim(C_TILE_SIZE), window_size),
                             dtype=dtype,
                             buffer=nl.psum)
        filt_psum[i_p, i_f_win] = nisa.tensor_copy(
            filt_sbuf[i_p, i_f_win],
            engine=nisa.vector_engine
        )

        # --- Phase B + Phase C (optimized): block the big 1D output dim ------
        # We tile the out_image_size dimension into fixed-size blocks.
        OUT_TILE = 64
        NUM_FULL_BLOCKS = out_image_size // OUT_TILE
        TAIL = out_image_size - NUM_FULL_BLOCKS * OUT_TILE

        for n in nl.affine_range(N):
            # prefetch this (n, c_tile) image slice into SBUF (same as original)
            img_local = nl.zeros((nl.par_dim(C_TILE_SIZE), image_size),
                                 dtype=dtype,
                                 buffer=nl.sbuf)
            i_w = nl.arange(W)[None, :]
            i_cin = i_p + base_chan
            i_img = W_padding_l + i_w
            img_local[i_p, i_img] = nl.load(
                img_ref[n, i_cin, 0, i_w]
            )

            # accumulate output per-(n, c_tile) in SBUF
            out_sb = nl.zeros((nl.par_dim(C_TILE_SIZE), out_image_size),
                              dtype=dtype,
                              buffer=nl.sbuf)

            # Process full blocks of OUT_TILE outputs
            for b in nl.affine_range(NUM_FULL_BLOCKS):
                base_out = b * OUT_TILE
                for o in nl.affine_range(OUT_TILE):
                    # identical to original per-output compute, now grouped by blocks
                    img_tile  = img_local[i_p, i_f_a + (base_out + o)]   # (C_TILE_SIZE, W_f)
                    filt_tile = filt_psum[i_p, i_f_win]                  # (C_TILE_SIZE, window_size)
                    prod = nisa.tensor_tensor(img_tile,
                                              filt_tile,
                                              op=np.multiply)
                    out_sb[i_p, base_out + o] = nisa.tensor_reduce(np.add,
                                                                   prod,
                                                                   axis=[1])

            # Handle tail outputs (if any) with the original scalar loop body
            if TAIL > 0:
                base_out = NUM_FULL_BLOCKS * OUT_TILE
                for o in nl.affine_range(TAIL):
                    idx = base_out + o
                    img_tile  = img_local[i_p, i_f_a + idx]             # (C_TILE_SIZE, W_f)
                    filt_tile = filt_psum[i_p, i_f_win]                  # (C_TILE_SIZE, window_size)
                    prod = nisa.tensor_tensor(img_tile,
                                              filt_tile,
                                              op=np.multiply)
                    out_sb[i_p, idx] = nisa.tensor_reduce(np.add,
                                                          prod,
                                                          axis=[1])

            # --- Phase C: write back to HBM (same structure as original) -----
            pred = (base_chan + i_p) < C_out
            for r in nl.affine_range(K0):
                base = r * K1
                row = out_sb[i_p, base + i_f_k1]
                nl.store(out_hbm[n,
                                 base_chan:base_chan + C_TILE_SIZE,
                                 r,
                                 0:K1],
                         row,
                         mask=pred)

    return out_hbm
