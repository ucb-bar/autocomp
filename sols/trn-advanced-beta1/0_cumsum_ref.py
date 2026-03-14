@nki.jit
def test(x, axis=None, p_size=None, f_size=None, acc_dtype=None):
    assert isinstance(axis, int) or axis is None
    if axis is None:
        axis = -1

    rank = x.ndim
    axis = normalize_dim(axis, rank)
    assert axis == rank - 1, "Only support cusum over last dim"

    x_shape = x.shape
    shape_2d = (n_elts(x_shape[:-1]), x_shape[-1])
    x = x.reshape(shape_2d)

    # Create output tensor in HBM with same shape as x (dtype matches input)
    y = nl.ndarray(shape_2d, dtype=x.dtype, buffer=nl.shared_hbm)

    pmax = nl.tile_size.pmax if p_size is None else p_size
    f_tile_size = 2048 if f_size is None else f_size

    pi, fi = nl.mgrid[0:pmax, 0:f_tile_size]

    acc_dtype = acc_dtype or x.dtype

    ones = nl.ones((pmax, f_tile_size), dtype=acc_dtype)

    for i in nl.affine_range(div_ceil(shape_2d[0], pmax)):
        n_f_tiles = div_ceil(shape_2d[1], f_tile_size)
        init = nl.zeros((pmax, 1), dtype=acc_dtype)

        for j in nl.sequential_range(n_f_tiles):
            mask = (i * pmax + pi < shape_2d[0]) & (j * f_tile_size + fi < shape_2d[1])
            data = nl.load(x[i * pmax + pi, j * f_tile_size + fi], mask=mask)

            result = nisa.tensor_tensor_scan(
                data0=ones, data1=data, initial=init,
                op0=np.multiply, op1=np.add,
                dtype=acc_dtype, mask=mask
            )

            nl.store(y[i * pmax + pi, j * f_tile_size + fi], result, mask=mask)

            # Carry the last value to the next tile (skipped for final tile)
            init[:, :] = nl.copy(result[:, f_tile_size - 1], mask=j + 1 < n_f_tiles)

    return y.reshape(x_shape)
