import numpy as np
import nki.language as nl
from operator import mul
from functools import reduce
import nki.isa as nisa
import nki
import torch
from torch_xla.core import xla_model as xm

# SUBSTITUTE HERE

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

def test_nki(ref_func, test_func):
    device = xm.xla_device()
    for _ in range(2):
        x_np = np.random.rand(2048, 2048).astype(np.float32)
        x = torch.from_numpy(x_np).to(device=device)
        ref_y = ref_func(x)
        test_y = test_func(x)
        if not np.allclose(test_y.detach().cpu().numpy(), ref_y.detach().cpu().numpy(), atol=1e-4, rtol=1e-2):
            return False
    return True

def benchmark_nki(nki_func):
    device = xm.xla_device()
    x_np = np.random.rand(2048,2048).astype(np.float32)
    x = torch.from_numpy(x_np).to(device=device)
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