def div_ceil(n, d):
    return (n + d - 1) // d


@nki.jit
def solution(img_ref, filter_ref, padding=None):
    W_padding_l, W_padding_r = padding[1]

    N, C_in, H, W = img_ref.shape
    C_out, _, H_f, W_f = filter_ref.shape

    K0 = H - H_f + 1
    K1 = W + W_padding_l + W_padding_r - W_f + 1
    padded_W = W + W_padding_l + W_padding_r
    dtype = img_ref.dtype

    out_hbm = nl.ndarray((N, C_out, K0, K1), dtype=dtype, buffer=nl.shared_hbm)

    C_NUM_TILES = div_ceil(C_in, 128)
    C_TILE_SIZE = min(C_in, 128)

    for i_n in nl.affine_range(N):
        for c_tile in nl.affine_range(C_NUM_TILES):
            c_start = c_tile * C_TILE_SIZE
            c_end = min(c_start + C_TILE_SIZE, C_in)
            tile_c = c_end - c_start

            # P=C_TILE_SIZE=128, F=padded_W: load padded image row
            img_local = nl.ndarray((C_TILE_SIZE, padded_W), dtype=dtype, buffer=nl.sbuf)
            nisa.memset(dst=img_local, value=0.0)
            img_row = nl.ndarray((C_TILE_SIZE, W), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=img_row[0:tile_c, 0:W],
                          src=img_ref[i_n, c_start:c_end, 0, 0:W])
            nisa.tensor_copy(dst=img_local[0:tile_c, W_padding_l:W_padding_l+W],
                             src=img_row[0:tile_c, 0:W])

            # P=C_TILE_SIZE=128, F=W_f: load filter row
            filt_local = nl.ndarray((C_TILE_SIZE, W_f), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=filt_local[0:tile_c, 0:W_f],
                          src=filter_ref[c_start:c_end, 0, 0, 0:W_f])

            for i_out in nl.affine_range(K1):
                prod = nl.ndarray((C_TILE_SIZE, W_f), dtype=dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=prod[0:tile_c, 0:W_f],
                                   data1=img_local[0:tile_c, i_out:i_out+W_f],
                                   data2=filt_local[0:tile_c, 0:W_f],
                                   op=nl.multiply)
                out = nl.ndarray((C_TILE_SIZE, 1), dtype=dtype, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=out[0:tile_c, 0:1],
                                   op=nl.add, data=prod[0:tile_c, 0:W_f],
                                   axis=1, keepdims=True)
                nisa.dma_copy(dst=out_hbm[i_n, c_start:c_end, 0, i_out:i_out+1],
                              src=out[0:tile_c, 0:1])

    return out_hbm
