def div_ceil(n, d):
    return (n + d - 1) // d


def get_3d_shape(ref, dim):
    before = 1
    for s in ref.shape[:dim]:
        before *= s
    after = 1
    for s in ref.shape[dim+1:]:
        after *= s
    return [before, ref.shape[dim], after]


@nki.jit
def test(ref, dim):
    assert len(ref.shape) >= 2
    assert dim != len(ref.shape) - 1

    ref = ref.reshape(get_3d_shape(ref, dim))
    transposed_shape = (ref.shape[0], ref.shape[2], ref.shape[1])
    dst = nl.ndarray(shape=transposed_shape, buffer=nl.shared_hbm, dtype=ref.dtype)

    D0, B, N = ref.shape
    B_tile_size = min(128, B)
    N_tile_size = min(128, N)
    B_num_tiles = div_ceil(B, B_tile_size)
    N_num_tiles = div_ceil(N, N_tile_size)

    for d0 in nl.affine_range(D0):
        for b_out_tile in nl.affine_range(B_num_tiles):
            b_start = b_out_tile * B_tile_size
            b_end = min(b_start + B_tile_size, B)
            tile_b = b_end - b_start

            for n_out_tile in nl.affine_range(N_num_tiles):
                n_start = n_out_tile * N_tile_size
                n_end = min(n_start + N_tile_size, N)
                tile_n = n_end - n_start

                _local = nl.ndarray(shape=(B_tile_size, N_tile_size),
                                    dtype=ref.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=_local[0:tile_b, 0:tile_n],
                              src=ref[d0, b_start:b_end, n_start:n_end])

                transposed_psum = nl.ndarray(shape=(N_tile_size, B_tile_size),
                                             dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=transposed_psum[0:tile_n, 0:tile_b],
                                  data=_local[0:tile_b, 0:tile_n])

                transposed_local = nl.ndarray(shape=(N_tile_size, B_tile_size),
                                              dtype=ref.dtype, buffer=nl.sbuf)
                nisa.memset(dst=transposed_local, value=0.0)
                nisa.tensor_copy(dst=transposed_local[0:tile_n, 0:tile_b],
                                 src=transposed_psum[0:tile_n, 0:tile_b])

                nisa.dma_copy(dst=dst[d0, n_start:n_end, b_start:b_end],
                              src=transposed_local[0:tile_n, 0:tile_b])

    return dst
