@nki.jit
def test(A_DRAM, B_DRAM, TILES_IN_BLOCK_K=8, TILES_IN_BLOCK_M=4, TILES_IN_BLOCK_N=4):
  """
  Optimized matrix multiplication kernel

   Args:

      A_DRAM: an input tensor of shape [K, M], where K is a multiple of 1024
      and M is a multiple of 512.  It is the left-hand-side argument of the
      matrix multiplication, delivered transposed for optimal performance.

      B_DRAM: an input tensor of shape [K, N],  where K is a multiple of 1024
        and N is a multiple of 2048.  It is the right-hand-side argument of
        the matrix multiplication.

      Z_DRAM: the resulting output tensor of shape [M, N]

  """
  K, M = A_DRAM.shape
  _, N = B_DRAM.shape

  Z_DRAM = nl.ndarray([M, N], dtype=A_DRAM.dtype, buffer=nl.shared_hbm)

  TILE_K = nl.tile_size.pmax
  TILE_M = nl.tile_size.gemm_stationary_fmax
  TILE_N = nl.tile_size.gemm_moving_fmax

  NUM_BLOCK_K = K // (TILES_IN_BLOCK_K * TILE_K)
  NUM_BLOCK_M = M // (TILES_IN_BLOCK_M * TILE_M)
  NUM_BLOCK_N = N // (TILES_IN_BLOCK_N * TILE_N)

  assert NUM_BLOCK_K * TILES_IN_BLOCK_K * TILE_K == K
  assert NUM_BLOCK_M * TILES_IN_BLOCK_M * TILE_M == M
  assert NUM_BLOCK_N * TILES_IN_BLOCK_N * TILE_N == N

  for n2 in nl.affine_range(NUM_BLOCK_N):
    for m2 in nl.affine_range(NUM_BLOCK_M):

      # Partition Z and then ensure that we are Z-block stationary
      # This way, no matter how large K, M, and N are, Z is never spilled/loaded
      # We only need to store once
      Z_SBUF = nl.zeros((TILES_IN_BLOCK_M, nl.par_dim(TILE_M), TILES_IN_BLOCK_N * TILE_N), dtype=Z_DRAM.dtype, buffer=nl.sbuf)

      for k2 in nl.affine_range(NUM_BLOCK_K):
        A_SBUF = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), TILES_IN_BLOCK_M * TILE_M), dtype=A_DRAM.dtype, buffer=nl.sbuf)
        B_SBUF = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), TILES_IN_BLOCK_N * TILE_N), dtype=B_DRAM.dtype, buffer=nl.sbuf)

        # Load in a block of A and a block of B
        for k1 in nl.affine_range(TILES_IN_BLOCK_K):
          k_start = k2 * TILES_IN_BLOCK_K * TILE_K + k1 * TILE_K
          k_end = k_start + TILE_K

          m_start = m2 * TILES_IN_BLOCK_M * TILE_M
          m_end = m_start + TILES_IN_BLOCK_M * TILE_M

          n_start = n2 * TILES_IN_BLOCK_N * TILE_N
          n_end = n_start + TILES_IN_BLOCK_N * TILE_N

          # We coalesce memory accesses by loading TILES_IN_BLOCK_M * TILE_M
          # values of A at a time. We cannot coalesce across K because K gets
          # split across the partition dimension
          A_SBUF[k1] = nl.load(A_DRAM[k_start:k_end, m_start:m_end])

          # We coalesce memory accesses by loading TILES_IN_BLOCK_N * TILE_N
          # values of B at a time. We cannot coalesce across K because K gets
          # split across the partition dimension
          B_SBUF[k1] = nl.load(B_DRAM[k_start:k_end, n_start:n_end])

        for m1 in nl.affine_range(TILES_IN_BLOCK_M):
          for n1 in nl.affine_range(TILES_IN_BLOCK_N):
            # Keep the tile of Z stationary in the PSUM buffer to minimize the
            # number of calls to nl.loop_reduce
            Z_PSUM = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

            m_start = m1 * TILE_M
            m_end = m_start + TILE_M

            n_start = n1 * TILE_N
            n_end = n_start + TILE_N

            for k1 in nl.affine_range(TILES_IN_BLOCK_K):
              Z_PSUM += nisa.nc_matmul(A_SBUF[k1, :, m_start:m_end], B_SBUF[k1, :, n_start:n_end])

            Z_SBUF[m1, :, n_start:n_end] = nl.loop_reduce(Z_PSUM, op=np.add, loop_indices=[k2], dtype=Z_DRAM.dtype)

      for m1 in nl.affine_range(TILES_IN_BLOCK_M):
        m_start = m2 * TILES_IN_BLOCK_M * TILE_M + m1 * TILE_M
        m_end = m_start + TILE_M

        n_start = n2 * TILES_IN_BLOCK_N * TILE_N
        n_end = n_start + TILES_IN_BLOCK_N * TILE_N

        # We coalesce memory accesses by storing TILES_IN_BLOCK_N * TILE_N
        # values of Z at a time. We cannot coalesce across M because M gets
        # split across the partition dimension
        nl.store(Z_DRAM[m_start:m_end, n_start:n_end], value=Z_SBUF[m1])

  return Z_DRAM
