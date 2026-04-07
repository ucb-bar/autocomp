import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import math
import torch
from torch_xla.core import xla_model as xm

# SUBSTITUTE HERE

@nki.jit
def ref(lhsT, rhs):
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_

  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax                  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax      # 512

  assert K % TILE_K == 0
  assert M % TILE_M == 0
  assert N % TILE_N == 0

  lhsT_f16 = nl.ndarray((nl.par_dim(TILE_K), TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
  rhs_f16 = nl.ndarray((nl.par_dim(TILE_K), TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
  lhsT_bf = nl.ndarray((nl.par_dim(TILE_K), TILE_M), dtype=nl.bfloat16, buffer=nl.sbuf)
  rhs_bf = nl.ndarray((nl.par_dim(TILE_K), TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)

  tmp_psum = nl.ndarray((nl.par_dim(TILE_M), TILE_N), dtype=nl.float32, buffer=nl.psum)
  tmp_sbuf = nl.ndarray((nl.par_dim(TILE_M), TILE_N), dtype=nl.float32, buffer=nl.sbuf)
  acc_sbuf = nl.ndarray((nl.par_dim(TILE_M), TILE_N), dtype=nl.float32, buffer=nl.sbuf)
  out_sbuf = nl.ndarray((nl.par_dim(TILE_M), TILE_N), dtype=result.dtype, buffer=nl.sbuf)

  for m in nl.affine_range(M // TILE_M):
    m0 = m * TILE_M
    m1 = m0 + TILE_M
    for n in nl.affine_range(N // TILE_N):
      n0 = n * TILE_N
      n1 = n0 + TILE_N

      nisa.memset(dst=acc_sbuf, value=0.0)
      for k in nl.affine_range(K // TILE_K):
        k0 = k * TILE_K

        nisa.dma_copy(dst=lhsT_f16, src=lhsT[k0:k0+TILE_K, m0:m1])
        nisa.dma_copy(dst=rhs_f16,  src=rhs[k0:k0+TILE_K, n0:n1])

        nisa.tensor_copy(dst=lhsT_bf, src=lhsT_f16, engine=nisa.vector_engine, dtype=nl.bfloat16)
        nisa.tensor_copy(dst=rhs_bf,  src=rhs_f16,  engine=nisa.vector_engine, dtype=nl.bfloat16)

        tmp_psum = nl.ndarray((nl.par_dim(TILE_M), TILE_N), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=tmp_psum, stationary=lhsT_bf, moving=rhs_bf)

        tmp_sbuf = nl.ndarray((nl.par_dim(TILE_M), TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tmp_sbuf, src=tmp_psum, engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=acc_sbuf, data1=acc_sbuf, data2=tmp_sbuf, op=nl.add, engine=nisa.vector_engine)

      nisa.tensor_copy(dst=out_sbuf, src=acc_sbuf, engine=nisa.vector_engine)
      nisa.dma_copy(dst=result[m0:m1, n0:n1], src=out_sbuf)

  return result


def test_nki(ref_func, test_func):
    device = xm.xla_device()
    for _ in range(3):
        a_np = np.random.randn(512, 512) * (1.0 / np.sqrt(512))
        b_np = np.random.randn(512, 512) * (1.0 / np.sqrt(512))
        a = torch.tensor(a_np, dtype=torch.bfloat16, device=device)
        b = torch.tensor(b_np, dtype=torch.bfloat16, device=device)
        result_1 = ref_func(a, b)
        result_2 = test_func(a, b)
        r1 = result_1.detach().cpu().to(torch.float32).numpy()
        r2 = result_2.detach().cpu().to(torch.float32).numpy()
        print("result_1", r1[:5, :5])
        print("result_2", r2[:5, :5])
        if not np.allclose(r1, r2, atol=1e-2, rtol=1e-3):
            return False
    return True

def benchmark_nki(nki_func):
    device = xm.xla_device()
    lhsT_np = np.random.randn(8192, 4096) * (1.0 / np.sqrt(4096))
    rhs_np = np.random.randn(8192, 8192) * (1.0 / np.sqrt(8192))
    lhsT = torch.tensor(lhsT_np, dtype=torch.bfloat16, device=device)
    rhs = torch.tensor(rhs_np, dtype=torch.bfloat16, device=device)

    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(lhsT, rhs)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))

if __name__ == "__main__":
    test_result = test_nki(ref, solution)
    if not test_result:
        print("Test failed")
        exit(1)
    else:
        print("Test passed")
        benchmark_nki(solution)
