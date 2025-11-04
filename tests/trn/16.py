import numpy as np
import neuronxcc.nki.language as nl
from operator import mul
from functools import reduce
import neuronxcc.nki.isa as nisa
import neuronxcc.nki as nki

# SUBSTITUTE HERE

def div_ceil(n, d):
    return (n + d - 1) // d

def normalize_dim(idx, rank):
    return idx if idx >= 0 else (rank + idx)

def n_elts(shape):
    return reduce(mul, shape, 1)

@nki.jit
def ref(x, axis=None, p_size=None, f_size=None, acc_dtype=None):
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

def test_nki(ref_func, test_func):
    for _ in range(2):
        x = np.random.rand(2048, 2048).astype(np.float32)
        y = np.zeros((10, 10), dtype=np.float32)
        ref_y = ref_func(x)
        test_y = test_func(x)
        if not np.allclose(test_y, ref_y,atol=1e-4,rtol=1e-2):
            return False
    return True

def benchmark_nki(nki_func):
    x = np.random.rand(2048,2048).astype(np.float32)
    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    _ = bench_func(x)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))
  
if __name__ == "__main__":
    # benchmark_nki(ref)
    test_result = test_nki(ref, test)
    if not test_result:
        print("Test failed")
        exit(1)
    else:
        print("Test passed")
        benchmark_nki(test)