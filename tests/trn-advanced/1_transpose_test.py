import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim
import neuronxcc.nki.language.constants as constants

# SUBSTITUTE HERE

def div_ceil(n, d):
  return (n + d - 1) // d

def get_3d_shape(ref, dim):
  new_shape = [int(np.prod(ref.shape[:dim])),
                ref.shape[dim],
                int(np.prod(ref.shape[dim+1:]))]
  return new_shape
@nki.jit
def ref(ref, dim):
  assert len(ref.shape) >= 2
  assert dim != len(ref.shape) - 1

  ref = ref.reshape(get_3d_shape(ref, dim))
  transposed_shape = (ref.shape[0], ref.shape[2], ref.shape[1])
  dst = nl.ndarray(shape=transposed_shape, buffer=nl.hbm, dtype=ref.dtype)
  transpose_nonlocal = dst.reshape(transposed_shape)

  D0, B, N = ref.shape
  B_tile_size = min(128, B)
  N_tile_size = min(128, N)
  B_num_tiles = div_ceil(B, B_tile_size)
  N_num_tiles = div_ceil(N, N_tile_size)
  for d0 in nl.affine_range(D0):
    for b_out_tile in nl.affine_range(B_num_tiles):
      for n_out_tile in nl.affine_range(N_num_tiles):
        _local = nl.ndarray(shape=(B_tile_size, N_tile_size), 
                                      dtype=ref.dtype, buffer=nl.sbuf, name='local')
        transposed_local = nl.ndarray(shape=(par_dim(N_tile_size), B_tile_size), 
                                      dtype=ref.dtype, buffer=nl.sbuf, name='transposed_local')
        i = nl.arange(0, B_tile_size)[:, None]
        j = nl.arange(0, N_tile_size)[None, :]
        mask = (b_out_tile * B_tile_size + i < B) & (n_out_tile * N_tile_size + j < N)
        #TODO: maybe better performance by refetching the ref tensor
        _local[i, j] = nl.load(ref[d0, b_out_tile * B_tile_size + i, n_out_tile * N_tile_size + j], mask=mask)

        p = nl.arange(0, N_tile_size)[:, None]
        q = nl.arange(0, B_tile_size)[None, :]
        transposed_local[p, q] = nisa.nc_transpose(_local[i, j], mask=mask)

        mask = (b_out_tile * B_tile_size + q < B) & (n_out_tile * N_tile_size + p < N)
        nl.store(transpose_nonlocal[d0, n_out_tile * N_tile_size + p, b_out_tile * B_tile_size + q], transposed_local[p, q], mask=mask)
  return dst

def test_nki(ref_func, test_func):
    for _ in range(2):
        x = np.random.rand(512, 512, 512).astype(np.float32)
        ref_out = ref_func(x,1)
        test_out = test_func(x,1)
        if not np.allclose(test_out, ref_out,atol=1e-4,rtol=1e-2):
            return False
    return True

def benchmark_nki(nki_func):
    x = np.random.rand(512, 512, 512).astype(np.float32)
    dim = 1
    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(x,1)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))

if __name__ == "__main__":
   
    test_result = test_nki(ref, test)
    if test_result:
        print("Test passed")
        benchmark_nki(test)
    else:
        print("Test failed")
        exit(1)