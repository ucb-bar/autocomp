import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


# SUBSTITUTE HERE

@nki.jit
def ref(a_input, b_input):
  """NKI kernel to compute element-wise addition of two input tensors

  Args:
      a_input: a first input tensor
      b_input: a second input tensor

  Returns:
      c_output: an output tensor
  """
  # Create output tensor
  c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)

  # Process in tiles of 128x512 due to hardware limitations
  tile_size_x = 128
  tile_size_y = 512

  for i in range(0, a_input.shape[0], tile_size_x):
    for j in range(0, a_input.shape[1], tile_size_y):
      # Generate tensor indices for this tile
      ix = i + nl.arange(tile_size_x)[:, None]
      iy = j + nl.arange(tile_size_y)[None, :]

      # Load input data from device memory (HBM) to on-chip memory (SBUF)
      a_tile = nl.load(a_input[ix, iy])
      b_tile = nl.load(b_input[ix, iy])

      # compute a + b
      c_tile = a_tile + b_tile

      # store the addition results back to device memory (c_output)
      nl.store(c_output[ix, iy], value=c_tile)

  return c_output


def test_nki(ref_func, test_func):
  for _ in range(2):
    x = np.random.rand(256, 1024).astype(np.float16)
    y = np.random.rand(256, 1024).astype(np.float16)
    ref_out = ref_func(x, y)
    test_out = test_func(x, y)
    if not np.allclose(ref_out, test_out, atol=1e-4, rtol=1e-2):
      return False
    # assert np.allclose(ref_out, test_out, atol=1e-4, rtol=1e-2)
  return True

def benchmark_nki(nki_func):
  a = np.random.rand(256, 1024).astype(np.float16)
  b = np.random.rand(256, 1024).astype(np.float16)
  bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
  bench_func(a, b)
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
  # a = np.random.rand(256, 1024).astype(np.float16)
  # b = np.random.rand(256, 1024).astype(np.float16)

  # output_nki = nki_tensor_add(a, b)
  # print(f"output_nki={output_nki}")

  # output_np = a + b
  # print(f"output_np={output_np}")

  # allclose = np.allclose(output_np, output_nki, atol=1e-4, rtol=1e-2)
  # if allclose:
  #   print("NKI and NumPy match")
  # else:
  #   print("NKI and NumPy differ")

  # assert allclose