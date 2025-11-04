import math

from neuronxcc import nki
import neuronxcc.nki.language as nl
import numpy as np
import torch
import torch.nn.functional as F

# SUBSTITUTE HERE

@nki.jit
def ref(in_tensor: nki.tensor, pool_size: int) -> nki.tensor:
    """
    Performs 2D max pooling with stride 1 on a 2D tensor.
    
    Args:
        in_tensor: Input tensor with shape [height, width]
        pool_size: Size of the pooling window (pool_size x pool_size)
        
    Returns:
        Output tensor with shape [height-(pool_size-1), width-(pool_size-1)]
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


def test_nki(ref_func, test_func):
    for _ in range(2):
        H, W = 4096, 4096
        pool_size = 3
        
        input_tensor = np.random.rand(H, W).astype(np.float32)
        result_1 = ref_func(input_tensor, pool_size)
        result_2 = test_func(input_tensor, pool_size)
        if not np.allclose(result_1, result_2, atol=1e-4, rtol=1e-2):
            return False
    return True
    

def benchmark_nki(nki_func):
    H, W = 4096, 4096
    pool_size = 3
    
    input_tensor = np.random.rand(H, W).astype(np.float32)
    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(input_tensor, pool_size)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))
    

if __name__ == "__main__":
    test_result = test_nki(ref, test)
    if not test_result:
        print("Test failed")
        exit(1)
    else:
        print("Test passed")
        benchmark_nki(test)