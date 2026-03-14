import math
import numpy as np
from neuronxcc import nki
import neuronxcc.nki.language as nl


@nki.jit
def ref(in_tensor: nki.tensor, pool_size: int) -> nki.tensor:
    """
    Performs 2D max pooling with stride 1 on a 2D tensor.
    """
    k = pool_size
    h_in, w_in = in_tensor.shape
    h_out, w_out = h_in - (k-1), w_in - (k-1)
    out_tensor = nl.ndarray((h_out, w_out), dtype=in_tensor.dtype, buffer=nl.shared_hbm)

    h_tiles_count = math.ceil(h_in / nl.tile_size.pmax)
    for h_tile_idx in nl.affine_range(h_tiles_count):
        in_tile = nl.ndarray((nl.par_dim(nl.tile_size.pmax), k, w_in), dtype=in_tensor.dtype, buffer=nl.sbuf)
        i_h, i_kh, i_w = nl.mgrid[0:nl.tile_size.pmax, 0:k, 0:w_in]
        i_h = h_tile_idx * nl.tile_size.pmax + i_h
        in_tile = nl.load(in_tensor[i_h + i_kh, i_w], mask=(i_h < (h_in - (k-1))))
        i_h, i_kh, i_w, i_kw = nl.mgrid[0:nl.tile_size.pmax, 0:k, 0:(w_in - (k-1)), 0:k]
        out_tile = nl.max(in_tile[i_h, i_kh, i_w + i_kw], axis=[1, 3], mask=(h_tile_idx * nl.tile_size.pmax + i_h < h_in))
        i_h_out, i_w_out = nl.mgrid[0:nl.tile_size.pmax, 0:(w_in - (k-1))]
        i_h_out = h_tile_idx * nl.tile_size.pmax + i_h_out
        nl.store(out_tensor[i_h_out, i_w_out], value=out_tile, mask=(i_h_out < h_out))

    return out_tensor


if __name__ == "__main__":
    input_tensor = np.load("maxpool_input.npy")
    pool_size = int(np.load("maxpool_pool_size.npy"))
    out = ref(input_tensor, pool_size)
    np.save("out_beta1_maxpool.npy", out)
