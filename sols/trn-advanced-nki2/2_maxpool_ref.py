@nki.jit
def solution(in_tensor, pool_size):
    k = pool_size
    h_in, w_in = in_tensor.shape
    h_out = h_in - (k - 1)
    w_out = w_in - (k - 1)
    out_tensor = nl.ndarray((h_out, w_out), dtype=in_tensor.dtype, buffer=nl.shared_hbm)

    PMAX = nl.tile_size.pmax  # 128
    W_OUT_TILE = 512

    h_tiles = (h_out + PMAX - 1) // PMAX
    w_tiles = (w_out + W_OUT_TILE - 1) // W_OUT_TILE

    for h_tile_idx in nl.affine_range(h_tiles):
        h_start = h_tile_idx * PMAX
        tile_h = min(PMAX, h_out - h_start)

        for w_tile_idx in nl.affine_range(w_tiles):
            w_start = w_tile_idx * W_OUT_TILE
            tile_w_out = min(W_OUT_TILE, w_out - w_start)
            tile_w_in = tile_w_out + k - 1  # extra columns for the window

            # Allocate row_max buffer (size based on tile_w_in)
            row_max = nl.ndarray((PMAX, W_OUT_TILE + k - 1), dtype=in_tensor.dtype, buffer=nl.sbuf)

            # Load first strip (ki=0) as initial row_max
            strip0 = nl.ndarray((PMAX, W_OUT_TILE + k - 1), dtype=in_tensor.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=strip0[0:tile_h, 0:tile_w_in],
                          src=in_tensor[h_start:h_start+tile_h, w_start:w_start+tile_w_in])
            nisa.tensor_copy(dst=row_max[0:tile_h, 0:tile_w_in],
                             src=strip0[0:tile_h, 0:tile_w_in])

            # Load strips 1..k-1 and accumulate max
            if k > 1:
                strip1 = nl.ndarray((PMAX, W_OUT_TILE + k - 1), dtype=in_tensor.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=strip1[0:tile_h, 0:tile_w_in],
                              src=in_tensor[h_start+1:h_start+1+tile_h, w_start:w_start+tile_w_in])
                nisa.tensor_tensor(dst=row_max[0:tile_h, 0:tile_w_in],
                                   data1=row_max[0:tile_h, 0:tile_w_in],
                                   data2=strip1[0:tile_h, 0:tile_w_in],
                                   op=nl.maximum)
            if k > 2:
                strip2 = nl.ndarray((PMAX, W_OUT_TILE + k - 1), dtype=in_tensor.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=strip2[0:tile_h, 0:tile_w_in],
                              src=in_tensor[h_start+2:h_start+2+tile_h, w_start:w_start+tile_w_in])
                nisa.tensor_tensor(dst=row_max[0:tile_h, 0:tile_w_in],
                                   data1=row_max[0:tile_h, 0:tile_w_in],
                                   data2=strip2[0:tile_h, 0:tile_w_in],
                                   op=nl.maximum)

            # Column-wise max: initialize with shift 0, then max with shifts 1..k-1
            col_max = nl.ndarray((PMAX, W_OUT_TILE), dtype=in_tensor.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=col_max[0:tile_h, 0:tile_w_out],
                             src=row_max[0:tile_h, 0:tile_w_out])

            if k > 1:
                nisa.tensor_tensor(dst=col_max[0:tile_h, 0:tile_w_out],
                                   data1=col_max[0:tile_h, 0:tile_w_out],
                                   data2=row_max[0:tile_h, 1:1+tile_w_out],
                                   op=nl.maximum)
            if k > 2:
                nisa.tensor_tensor(dst=col_max[0:tile_h, 0:tile_w_out],
                                   data1=col_max[0:tile_h, 0:tile_w_out],
                                   data2=row_max[0:tile_h, 2:2+tile_w_out],
                                   op=nl.maximum)

            nisa.dma_copy(dst=out_tensor[h_start:h_start+tile_h, w_start:w_start+tile_w_out],
                          src=col_max[0:tile_h, 0:tile_w_out])

    return out_tensor
