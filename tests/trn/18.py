import math
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import torch_xla.core.xla_model as xm
import numpy as np

# SUBSTITUTE HERE

@nki.jit
def ref(a_tensor):
  # Calculate out_tensor
  # Where softmax(x) = = exp(x - max(x)) / sum(exp(x - max(x)))
  out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                          buffer=nl.shared_hbm)

  # Generate tensor indices to index input tensor
  ix = nl.arange(128)[:, None]
  iy = nl.arange(a_tensor.shape[1])[None, :]

  num_rows = a_tensor.shape[0]

  # Process 128 rows at a time due to 128-partition tile size limitation
  # Since we're not reducing across the first dimension
  # Tiles can be processed independently
  for i in nl.affine_range(math.ceil(a_tensor.shape[0]/128)):

    # Load input data from external memory to on-chip memory
    a_tile = nl.load(a_tensor[i * 128 + ix, iy],
                    mask=(i * 128 + ix < num_rows))

    # Find max and subtract from each value to ensure numerical stability
    max_vals = nl.max(a_tile, axis=[1], keepdims=True, mask=(i * 128 + ix < num_rows))
    shifted = nl.subtract(a_tile, max_vals, mask=(i * 128 + ix < num_rows))

    # Compute element-wise exp of a_tensor
    numerator = nl.exp(shifted)

    # Calculate sum of squared elements, along last dimension
    denominator = nl.sum(numerator, axis=[1])

    # Scale and get a reciprocal
    sm = numerator / denominator

    # store the results back to external memory (out_tensor)
    nl.store(out_tensor[i * 128 + ix, iy], value=sm,
            mask=(i * 128 + ix < num_rows))

  return out_tensor


def test_nki(ref_func, test_func):
    for _ in range(2):
        a_tensor = np.random.rand(1024, 1024).astype(np.float32)
        ref_out = ref_func(a_tensor)
        test_out = test_func(a_tensor)
        if not np.allclose(test_out, ref_out, atol=1e-4, rtol=1e-2):
            return False
    return True

def benchmark_nki(nki_func):
    a_tensor = np.random.rand(4096, 4096).astype(np.float32)
    # nki.benchmark(warmup=2, iters=10)(nki_func)(a_tensor)
    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(a_tensor)
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
