@nki.jit
def test(x, axis=None, p_size=None, f_size=None, acc_dtype=None):
    assert isinstance(axis, int) or axis is None
    if axis is None:
        axis = -1

    rank = len(x.shape)
    axis = normalize_dim(axis, rank)
    assert axis == rank - 1, "Only support cusum over last dim"

    x_shape = x.shape
    shape_2d = (n_elts(x_shape[:-1]), x_shape[-1])
    x = x.reshape(shape_2d)

    y = nl.ndarray(shape_2d, dtype=x.dtype, buffer=nl.shared_hbm)

    pmax = nl.tile_size.pmax if p_size is None else p_size
    f_tile_size = 2048 if f_size is None else f_size

    acc_dtype = acc_dtype or x.dtype

    zeros_buf = nl.ndarray((pmax, f_tile_size), dtype=acc_dtype, buffer=nl.sbuf)
    nisa.memset(dst=zeros_buf, value=0.0)
    ones = nl.ndarray((pmax, f_tile_size), dtype=acc_dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ones, data=zeros_buf, op0=nl.add, operand0=1.0)

    for i in nl.affine_range(div_ceil(shape_2d[0], pmax)):
        p_start = i * pmax
        p_end = min(p_start + pmax, shape_2d[0])
        tile_p = p_end - p_start

        n_f_tiles = div_ceil(shape_2d[1], f_tile_size)
        init = nl.ndarray((pmax, 1), dtype=acc_dtype, buffer=nl.sbuf)
        nisa.memset(dst=init[0:tile_p, 0:1], value=0.0)

        for j in nl.sequential_range(n_f_tiles):
            f_start = j * f_tile_size
            f_end = min(f_start + f_tile_size, shape_2d[1])
            tile_f = f_end - f_start

            data = nl.ndarray((pmax, f_tile_size), dtype=acc_dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=data[0:tile_p, 0:tile_f], src=x[p_start:p_end, f_start:f_end])

            result = nl.ndarray((pmax, f_tile_size), dtype=acc_dtype, buffer=nl.sbuf)
            nisa.tensor_tensor_scan(
                dst=result[0:tile_p, 0:tile_f],
                data0=ones[0:tile_p, 0:tile_f],
                data1=data[0:tile_p, 0:tile_f],
                initial=init[0:tile_p, 0:1],
                op0=nl.multiply, op1=nl.add,
            )

            nisa.dma_copy(dst=y[p_start:p_end, f_start:f_end], src=result[0:tile_p, 0:tile_f])

            # Carry last column to next tile
            nisa.tensor_copy(dst=init[0:tile_p, 0:1], src=result[0:tile_p, tile_f-1:tile_f])

    return y.reshape(x_shape)
