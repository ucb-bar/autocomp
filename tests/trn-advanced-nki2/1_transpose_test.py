import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
import torch
from torch_xla.core import xla_model as xm
import os


# SUBSTITUTE HERE

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
def ref(ref, dim):
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
                nisa.dma_copy(dst=_local,
                              src=ref[d0, b_start:b_end, n_start:n_end])

                transposed_psum = nl.ndarray(shape=(N_tile_size, B_tile_size),
                                             dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=transposed_psum,
                                  data=_local)

                transposed_local = nl.ndarray(shape=(N_tile_size, B_tile_size),
                                              dtype=ref.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=transposed_local, src=transposed_psum)

                nisa.dma_copy(dst=dst[d0, n_start:n_end, b_start:b_end],
                              src=transposed_local)

    return dst

def test_nki(ref_func, test_func):
    device = xm.xla_device()
    for _ in range(2):
        x_np = np.random.rand(512, 512, 512).astype(np.float32)
        x = torch.from_numpy(x_np).to(device=device)
        ref_out = ref_func(x, 1)
        test_out = test_func(x, 1)
        if not np.allclose(test_out.detach().cpu().numpy(), ref_out.detach().cpu().numpy(), atol=1e-4, rtol=1e-2):
            return False
    return True

def benchmark_nki(nki_func):
    device = xm.xla_device()
    x_np = np.random.rand(512, 512, 512).astype(np.float32)
    x = torch.from_numpy(x_np).to(device=device)
    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(x, 1)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))

if __name__ == "__main__":
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn1" 
    test_result = test_nki(ref, test)
    if test_result:
        print("Test passed")
        benchmark_nki(test)
    else:
        print("Test failed")
        exit(1)