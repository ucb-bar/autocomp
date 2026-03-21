"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

RMSNorm NKI kernel implementation.

"""

import numpy as np
# NKI_EXAMPLE_42_BEGIN
import math
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

# SUBSTITUTE HERE

def row_indices_ref(width):
  ip = nl.arange(1)[:, None]
  iy = nl.arange(width)[None, :]
  return ip, iy

def row_dst_ref(out_tensor, row_idx):
  ip, iy = row_indices_ref(out_tensor.shape[1])
  return out_tensor[row_idx + ip, iy]

def square_ref(a_tensor, out_tensor):
  assert a_tensor.shape[0] == 1
  ip, iy = row_indices_ref(a_tensor.shape[1])
  a_tile = nl.load(a_tensor[ip, iy])
  result = nisa.tensor_tensor(a_tile, a_tile, op=np.multiply)
  nl.store(out_tensor[ip, iy], value=result)

def mean_ref(a_tensor, out_tensor):
  assert a_tensor.shape[0] == 1
  ip, iy = row_indices_ref(a_tensor.shape[1])
  iw = nl.arange(1)[None, :]
  a_tile = nl.load(a_tensor[ip, iy])
  result = nisa.tensor_reduce(op=np.add, data=a_tile,
                              axis=1, keepdims=True) / a_tensor.shape[1]
  nl.store(out_tensor[ip, iw], value=result)

def rsqrt_ref(a_tensor, out_tensor):
  assert a_tensor.shape == (1, 1)
  ip, _ = row_indices_ref(1)
  iw = nl.arange(1)[None, :]
  a_tile = nl.load(a_tensor[ip, iw])
  result = nisa.activation(op=nl.rsqrt, data=a_tile)
  nl.store(out_tensor[ip, iw], value=result)

def col_multiply_ref(a_tensor, b_tensor, out_tensor):
  assert a_tensor.shape[0] == 1
  assert b_tensor.shape == (1, 1)
  ip, iy = row_indices_ref(a_tensor.shape[1])
  iw = nl.arange(1)[None, :]
  a_tile = nl.load(a_tensor[ip, iy])
  b_tile = nl.load(b_tensor[ip, iw])
  b_bcast = b_tile.broadcast_to(a_tensor.shape)
  result = nisa.tensor_tensor(a_tile, b_bcast, op=np.multiply)
  nl.store(out_tensor[ip, iy], value=result)

def row_multiply_ref(a_tensor, g_tensor, out_tensor):
  assert a_tensor.shape[0] == 1
  ip, iy = row_indices_ref(a_tensor.shape[1])
  a_tile = nl.load(a_tensor[ip, iy])
  g_tile = nl.load(g_tensor.reshape((1, g_tensor.shape[0]))[ip, iy])
  result = nisa.tensor_tensor(a_tile, g_tile, op=np.multiply)
  nl.store(out_tensor[ip, iy], value=result)

def write_row_ref(out_tensor, row_idx, row_tensor):
  ip, iy = row_indices_ref(row_tensor.shape[1])
  nl.store(row_dst_ref(out_tensor, row_idx), value=nl.load(row_tensor[ip, iy]))

@nki.jit
def ref(a_tensor, g_tensor):
  """RMSNorm with per-row 1D helper functions, shared_hbm pre-allocated."""
  N = a_tensor.shape[1]

  out_tensor   = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
  squared_buf  = nl.ndarray((1, N), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
  mean_buf     = nl.ndarray((1, 1), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
  rsqrt_buf    = nl.ndarray((1, 1), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
  scaled_buf   = nl.ndarray((1, N), dtype=a_tensor.dtype, buffer=nl.shared_hbm)
  out_row_buf  = nl.ndarray((1, N), dtype=a_tensor.dtype, buffer=nl.shared_hbm)

  for i in nl.sequential_range(a_tensor.shape[0]):
    a_row = a_tensor[i:i + 1, :]
    square_ref(a_row, squared_buf)
    mean_ref(squared_buf, mean_buf)
    rsqrt_ref(mean_buf, rsqrt_buf)
    col_multiply_ref(a_row, rsqrt_buf, scaled_buf)
    row_multiply_ref(scaled_buf, g_tensor, out_row_buf)
    write_row_ref(out_tensor, i, out_row_buf)

  return out_tensor

def test_nki(ref_func, test_func):
    for _ in range(3):
        a = np.random.rand(512, 4096).astype(np.float32)
        g = np.random.rand(4096).astype(np.float32)
        result_1 = ref_func(a, g)
        result_2 = test_func(a, g)
        if not np.allclose(result_1, result_2, atol=1e-4, rtol=1e-2):
            return False
    return True

def benchmark_nki(nki_func):
    a = np.random.rand(512, 4096).astype(np.float32)
    g = np.random.rand(4096).astype(np.float32)

    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(a, g)
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