@nki.jit
def test(lhsT, rhs):
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

