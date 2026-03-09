# DEPRECATED
# import neuronxcc.nki as nki
# import neuronxcc.nki.isa as nisa
# import neuronxcc.nki.language as nl
# import neuronxcc.nki.typing as nt

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import math

# SUBSTITUTE HERE

@nki.jit
def nki_matmul_block_free_dimension_(lhsT, rhs):
  # Compute result = lhsT.T @ rhs, where lhsT is [K, M] and rhs is [K, N].
  # Accumulate in fp32; store output in the original dtype.
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

  # Tiles and scratch buffers.
  # Some releases have correctness issues on fp16 TensorE paths; use bf16 inputs for matmul.
  lhsT_f16 = nl.ndarray((nl.par_dim(TILE_K), TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)        # [K, M]
  rhs_f16 = nl.ndarray((nl.par_dim(TILE_K), TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)          # [K, N]
  lhsT_bf = nl.ndarray((nl.par_dim(TILE_K), TILE_M), dtype=nl.bfloat16, buffer=nl.sbuf)        # [K, M]
  rhs_bf = nl.ndarray((nl.par_dim(TILE_K), TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)         # [K, N]

  tmp_psum = nl.ndarray((nl.par_dim(TILE_M), TILE_N), dtype=nl.float32, buffer=nl.psum)        # [M, N] (PSUM)
  tmp_sbuf = nl.ndarray((nl.par_dim(TILE_M), TILE_N), dtype=nl.float32, buffer=nl.sbuf)        # [M, N]
  acc_sbuf = nl.ndarray((nl.par_dim(TILE_M), TILE_N), dtype=nl.float32, buffer=nl.sbuf)        # [M, N]
  out_sbuf = nl.ndarray((nl.par_dim(TILE_M), TILE_N), dtype=result.dtype, buffer=nl.sbuf)      # [M, N] casted

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

        # Allocate fresh PSUM tile each iteration
        tmp_psum = nl.ndarray((nl.par_dim(TILE_M), TILE_N), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=tmp_psum, stationary=lhsT_bf, moving=rhs_bf)

        tmp_sbuf = nl.ndarray((nl.par_dim(TILE_M), TILE_N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tmp_sbuf, src=tmp_psum, engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=acc_sbuf, data1=acc_sbuf, data2=tmp_sbuf, op=nl.add, engine=nisa.vector_engine)

      nisa.tensor_copy(dst=out_sbuf, src=acc_sbuf, engine=nisa.vector_engine)
      nisa.dma_copy(dst=result[m0:m1, n0:n1], src=out_sbuf)

  return result


''' DEPRECATED
@nki.jit
def nki_matmul_block_free_dimension_(lhsT, rhs):
  """NKI kernel to compute a matrix multiplication operation while blocking the
     free dimensions of the LHS and RHS to improve memory access pattern.

  Args:
      lhsT: an input tensor of shape [K,M], where both K and M are multiples for
        128.  It is the left-hand-side argument of the matrix multiplication,
        delivered transposed for optimal performance.
      rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
        is a multiple of 512.  It is the right-hand-side argument of the matrix
        multiplication.
  Returns:
      result: the resulting output tensor of shape [M,N]
  """

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # Define the indices (shape) of the tiles
  i_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
  i_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
  i_res = nl.mgrid[0:TILE_M, 0:TILE_N]

  # Configuring the blocking size for the free dimensions
  TILES_IN_BLOCK_M = 2
  TILES_IN_BLOCK_N = 2

  BLOCK_M = TILE_M * TILES_IN_BLOCK_M  # 256
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N  # 1024

  # the size has to be multiple of block size
  assert M % BLOCK_M == 0
  assert N % BLOCK_N == 0

  # Loop over blocks over the M dimension
  for m in nl.affine_range(M // BLOCK_M):
    # Load TILES_IN_BLOCK_M columns tiles from lhsT
    lhsT_tiles = nl.ndarray(
        (TILES_IN_BLOCK_M, K // TILE_K, nl.par_dim(TILE_K), TILE_M),
        dtype=lhsT.dtype,
        buffer=nl.sbuf)
    for bm in nl.affine_range(TILES_IN_BLOCK_M):
      for k in nl.affine_range(K // TILE_K):
        lhsT_tiles[bm, k, i_lhsT.p, i_lhsT.x] = nl.load(
            lhsT[k * TILE_K + i_lhsT.p,
                 (m * TILES_IN_BLOCK_M + bm) * TILE_M + i_lhsT.x])

    for n in nl.affine_range(N // BLOCK_N):
      # Load TILES_IN_BLOCK_N columns from rhs
      rhs_tiles = nl.ndarray(
          (TILES_IN_BLOCK_N, K // TILE_K, nl.par_dim(TILE_K), TILE_N),
          dtype=rhs.dtype,
          buffer=nl.sbuf)
      for bn in nl.affine_range(TILES_IN_BLOCK_N):
        for k in nl.affine_range(K // TILE_K):
          rhs_tiles[bn, k, i_rhs.p, i_rhs.x] = nl.load(
              rhs[k * TILE_K + i_rhs.p,
                  (n * TILES_IN_BLOCK_N + bn) * TILE_N + i_rhs.x])

      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          # Allocate a tensor in PSUM
          res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
          for k in nl.affine_range(K // TILE_K):
            # Accumulate partial-sums into PSUM
            res_psum += nl.matmul(lhsT_tiles[bm, k, i_lhsT.p, i_lhsT.x],
                                  rhs_tiles[bn, k, i_rhs.p, i_rhs.x],
                                  transpose_x=True)

          # Copy the result from PSUM back to SBUF, and cast to expected output data-type
          res_sb = nl.copy(res_psum, dtype=result.dtype)
          nl.store(result[(m * TILES_IN_BLOCK_M + bm) * TILE_M + i_res.p,
                          (n * TILES_IN_BLOCK_N + bn) * TILE_N + i_res.x],
                   value=res_sb)

  return result
'''

def test_nki(ref_func, test_func):
    for _ in range(3):
        # a = input_matrix = np.random.rand(8192, 4096).astype(nl.bfloat16)
        # b = input_matrix = np.random.rand(8192, 8192).astype(nl.bfloat16)
        a = (np.random.randn(8192, 4096) * (1.0 / np.sqrt(4096))).astype(nl.bfloat16)
        b = (np.random.randn(8192, 8192) * (1.0 / np.sqrt(8192))).astype(nl.bfloat16)
        result_1 = ref_func(a, b)
        result_2 = test_func(a, b)
        print("result_1", result_1[:5, :5])
        print("result_2", result_2[:5, :5])
        if not np.allclose(result_1.astype(nl.float32), result_2.astype(nl.float32), atol=1e-2, rtol=1e-3):
            return False
    return True

def benchmark_nki(nki_func):
    # Benchmarking with large matrices to show the differences more clearly
    lhsT = nt.tensor[[8192, 4096], nl.bfloat16]
    rhs = nt.tensor[[8192, 8192], nl.bfloat16]

    bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
    bench_func(lhsT, rhs)
    latency_res = bench_func.benchmark_result.nc_latency
    p99 = latency_res.get_latency_percentile(99)
    print("Latency: {:.2f} ms (P99)".format(p99 / 1000.0))

if __name__ == "__main__":
    test_result = test_nki(nki_matmul_block_free_dimension_, test)
    if not test_result:
        print("Test failed")
        exit(1)
    else:
        benchmark_nki(test)