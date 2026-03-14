import math

import nki
import nki.language as nl
import nki.isa as nisa
import numpy as np

# SUBSTITUTE HERE

@nki.jit
def ref(in_tensor, pool_size):
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
            tile_w_in = tile_w_out + k - 1

            row_max = nl.ndarray((PMAX, W_OUT_TILE + k - 1), dtype=in_tensor.dtype, buffer=nl.sbuf)

            strip0 = nl.ndarray((PMAX, W_OUT_TILE + k - 1), dtype=in_tensor.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=strip0[0:tile_h, 0:tile_w_in],
                          src=in_tensor[h_start:h_start+tile_h, w_start:w_start+tile_w_in])
            nisa.tensor_copy(dst=row_max[0:tile_h, 0:tile_w_in],
                             src=strip0[0:tile_h, 0:tile_w_in])

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