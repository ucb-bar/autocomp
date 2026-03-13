## fused_mamba.html

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np

@nki.jit
def mamba_v1(delta, u, A, B, C):
    """Computes the SSM operation in the Mamba model.

    :param delta: (batch_size, channels, seq_len)
    :param u: (batch_size, channels, seq_len)
    :param A: (channels, state_size)
    :param B: (batch_size, state_size, seq_len)
    :param C: (batch_size, state_size, seq_len)
    :return: (batch_size, channels, seq_len)
    """
    batch_size, channels, seq_len = delta.shape
    output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype,
                        buffer=nl.shared_hbm)

    _, state_size = A.shape

    # We can relax this using mask paramters in all the NKI API calls
    assert channels % 128 == 0

    # Map channels to the partition dimension
    # Tile channels to comply with NKI tile size constraints
    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize

    # Most outer loop with batch_size, parallel_for
    for i_batch in nl.affine_range(batch_size):
        # partial accumulated scanC result with processed states
        scanC_accum = nl.zeros((n_channel_tile, nl.par_dim(channel_psize), seq_len), dtype=delta.dtype)

        # Second outer loop with state_size, partial parallel
        for i_state in nl.affine_range(state_size):

            # Inner loop: tiling channels
            for i_channel_tile in nl.affine_range(n_channel_tile):
                channel_start = i_channel_tile * channel_psize

                # Load the relevant tile from delta and A
                delta_i = nl.load(delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
                A_i = nl.load(A[channel_start:channel_start+channel_psize, i_state])

                # Step 1&2: Element-wise multiplication of delta_i and A_i and then exponential
                deltaA = nisa.activation(op=nl.exp, data=delta_i, scale=A_i)

                # Load the relevant tile from u and B
                u_i = nl.load(u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
                B_i = nl.load(B[i_batch, i_state:i_state+1, 0:seq_len])

                # Step 3: Element-wise multiplication of delta_i, B_i and u_i
                deltaU = nisa.tensor_tensor(delta_i, u_i, op=nl.multiply)
                B_i_bcast = B_i.broadcast_to((channel_psize, seq_len))
                deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=nl.multiply)

                # Step 4: Associative scan between deltaA and deltaBu
                scan_res = nki.isa.tensor_tensor_scan(deltaA, deltaBu, initial=0,
                        op0=np.multiply, op1=np.add)

                # Load the relevant tile from C
                C_i = nl.load(C[i_batch, i_state:i_state+1, 0:seq_len])

                # Step 5: Element-wise multiplication of scan_res and C_i
                C_i_bcast = C_i.broadcast_to((channel_psize, seq_len))
                scanC = nisa.tensor_tensor(scan_res, C_i_bcast, op=nl.multiply)

                # Step 6: Accumulation of scanC along state_size dimension
                scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_len] += scanC

        # Store scanC_accum for a single batch to output
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize
            nl.store(output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
                    scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_len])

    return output
```

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np

@nki.jit
def mamba_v2(delta, u, A, B, C):
    """Computes the SSM operation in the Mamba model.

    :param delta: (batch_size, channels, seq_len)
    :param u: (batch_size, channels, seq_len)
    :param A: (channels, state_size)
    :param B: (batch_size, state_size, seq_len)
    :param C: (batch_size, state_size, seq_len)
    :return: (batch_size, channels, seq_len)
    """
    batch_size, channels, seq_len = delta.shape
    output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype,
                        buffer=nl.shared_hbm)
    _, state_size = A.shape

    assert channels % 128 == 0

    # Map channels to the partition dimension
    # Tile channels to comply with NKI tile size constraints
    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize

    # Most outer loop with batch_size, parallel_for
    for i_batch in nl.affine_range(batch_size):

        # Second outer loop: tiling channels
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize

            # partial accumulated scanC result with processed states
            scanC_accum = nl.zeros((nl.par_dim(channel_psize), seq_len), dtype=delta.dtype)

            # Load delta/u once to be reused across states
            delta_i = nl.load(delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
            u_i = nl.load(u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])

            # Inner loop with state_size, partial parallel
            for i_state in nl.affine_range(state_size):
                # Load the relevant tile from A
                A_i = nl.load(A[channel_start:channel_start+channel_psize, i_state])

                # Step 1&2: Element-wise multiplication of delta_i and A_i and then exponential
                deltaA = nisa.activation(op=nl.exp, data=delta_i, scale=A_i)

                # Load the relevant tile from B
                B_i = nl.load(B[i_batch, i_state:i_state+1, 0:seq_len])

                # Step 3: Element-wise multiplication of delta_i, B_i and u_i
                deltaU = nisa.tensor_tensor(delta_i, u_i, op=nl.multiply)
                B_i_bcast = B_i.broadcast_to((channel_psize, seq_len))
                deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=nl.multiply)

                # Step 4: Associative scan between deltaA and deltaBu
                scan_res = nki.isa.tensor_tensor_scan(deltaA, deltaBu, initial=0,
                        op0=np.multiply, op1=np.add)

                # Load the relevant tile from C
                C_i = nl.load(C[i_batch, i_state:i_state+1, 0:seq_len])

                # Step 5: Element-wise multiplication of scan_res and C_i
                C_i_bcast = C_i.broadcast_to((channel_psize, seq_len))
                scanC = nisa.tensor_tensor(scan_res, C_i_bcast, op=nl.multiply)

                # Step 6: Accumulation of scanC along state_size dimension
                scanC_accum[0:channel_psize, 0:seq_len] += scanC

            # Store scanC_accum for a single batch to output
            nl.store(output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
                    scanC_accum[0:channel_psize, 0:seq_len])

    return output
```

## matrix_multiplication.html

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa_helper as nisa

@nki.jit
def nki_matmul_basic_(lhsT, rhs):
  """NKI kernel to compute a 64x128x512 matrix multiplication operation

  Args:
      lhsT: an input tensor of shape [128,64], a left hand side argument of the
        matrix multiplication, delivered transposed for optimal performance
      rhs: an input tensor of shape [128,512], a right hand side argument of the
        matrix multiplication
  Returns:
      result: the resulting output tensor of shape [64,512]
  """
  result = nl.ndarray((64, 512), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Defining indexes for input LHS.T
  # - Note: here we take LayoutConstraint #1 into account:
  # "For MatMult, contraction axis must be mapped to P-dim"
  i_lhsT_p, i_lhsT_f = nl.mgrid[0:128, 0:64]

  # Defining indexes for input RHS
  # - Note: here we take LayoutConstraint #1 into account:
  # "For MatMult, contraction axis must be mapped to P-dim"
  i_rhs_p, i_rhs_f = nl.mgrid[0:128, 0:512]

  # Defining indexes for the output ([64,128]@[128,512] -> [64,512])
  i_out_p, i_out_f = nl.mgrid[0:64, 0:512]

  # Loading the inputs (HBM->SBUF)
  # Note: here we take Tile dtype definition into account,
  # which forces P-dim as the left most index
  lhs_tile = nl.load(lhsT[i_lhsT_p, i_lhsT_f])
  rhs_tile = nl.load(rhs[i_rhs_p, i_rhs_f])

  # Perform the matrix-multiplication
  # Note1: We set transpose_x to True, to indicate that the LHS input is transposed
  # Note2: A NKI matmul instruction always writes to PSUM in float32 data-type
  result_psum = nl.matmul(lhs_tile, rhs_tile, transpose_x=True)

  # Copy the result from PSUM back to SBUF, and cast to expected output data-type
  result_sbuf = nl.copy(result_psum, dtype=result.dtype)

  # The result of a [64,128] x [128,512] matrix multiplication has a shape of [64, 512].
  # This dictates which indices to use to address the result tile.
  nl.store(result[i_out_p, i_out_f], value=result_sbuf)

  return result
```

```python
@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
  """NKI kernel to compute a matrix multiplication operation in a tiled manner

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

  # Use affine_range to loop over tiles
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      # Allocate a tensor in PSUM
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

      for k in nl.affine_range(K // TILE_K):
        # Declare the tiles on SBUF
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

        # Load tiles from lhsT and rhs
        lhsT_tile[...] = nl.load(lhsT[k * TILE_K:(k + 1) * TILE_K,
                                      m * TILE_M:(m + 1) * TILE_M])
        rhs_tile[...] = nl.load(rhs[k * TILE_K:(k + 1) * TILE_K,
                                    n * TILE_N:(n + 1) * TILE_N])

        # Accumulate partial-sums into PSUM
        res_psum += nl.matmul(lhsT_tile[...], rhs_tile[...], transpose_x=True)

      # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.copy(res_psum, dtype=result.dtype)
      nl.store(result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N],
               value=res_sb)

  return result
```

```python
@nki.jit
def nki_matmul_hoist_load_(lhsT, rhs):
  """NKI kernel to compute a matrix multiplication operation in a tiled manner
     while hoisting the load of the lhsT and rhs to outer loops.

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

  # Use affine_range to loop over tiles
  for m in nl.affine_range(M // TILE_M):
    # Load a whole column tiles from lhsT (with K * TILE_N numbers)
    # This corresponds to the whole row in the original lhs
    lhsT_tiles = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), TILE_N),
                            dtype=lhsT.dtype,
                            buffer=nl.sbuf)

    for k in nl.affine_range(K // TILE_K):
      # use `.p` for partition dimension and `.x` for the first free dimension
      lhsT_tiles[k, i_lhsT.p, i_lhsT.x] = nl.load(lhsT[k * TILE_K + i_lhsT.p,
                                                       m * TILE_M + i_lhsT.x])

    for n in nl.affine_range(N // TILE_N):

      # Load a whole column tiles from rhs (with K * TILE_M numbers)
      rhs_tiles = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), TILE_N),
                             dtype=rhs.dtype,
                             buffer=nl.sbuf)
      for k in nl.affine_range(K // TILE_K):
        rhs_tiles[k, i_rhs.p, i_rhs.x] = nl.load(rhs[k * TILE_K + i_rhs.p,
                                                     n * TILE_N + i_rhs.x])

      # Allocate a tile in PSUM for the result
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
        # Accumulate partial-sums into PSUM
        res_psum[...] += nl.matmul(lhsT_tiles[k, i_lhsT.p, i_lhsT.x],
                                   rhs_tiles[k, i_rhs.p, i_rhs.x],
                                   transpose_x=True)

      # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.copy(res_psum, dtype=result.dtype)
      nl.store(result[m * TILE_M + i_res.p, n * TILE_N + i_res.x], value=res_sb)

  return result
```

```python
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
```

```python
@nki.jit
def nki_matmul_fully_optimized_(
    lhsT,
    rhs,
    # Meta-parameters
    TILES_IN_BLOCK_M=16,
    TILES_IN_BLOCK_N=2,
    TILES_IN_BLOCK_K=8,
):
  """NKI kernel to compute a large matrix multiplication efficiently by
     blocking all dimensions and doing layout optimization.

  Args:
      lhsT: an input tensor of shape [K,M], where K is a multiple of 128 *
        TILES_IN_BLOCK_K and M is a multiple of 128 * TILES_IN_BLOCK_M.  It is the
        left-hand-side argument of the matrix multiplication, delivered transposed
        for optimal performance.
      rhs: an input tensor of shape [K,N],  where K is a multiple of 128 *
        TILES_IN_BLOCK_K and N is a multiple of 512 * TILES_IN_BLOCK_N.  It is
        the right-hand-side argument of the matrix multiplication.
      TILES_IN_BLOCK_*: meta parameters to control blocking dimensions
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

  BLOCK_M = TILE_M * TILES_IN_BLOCK_M
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N
  BLOCK_K = TILE_K * TILES_IN_BLOCK_K

  # the size has to be multiple of block size
  assert M % BLOCK_M == 0
  assert N % BLOCK_N == 0
  assert K % BLOCK_K == 0

  NUM_BLOCK_M = M // BLOCK_M
  NUM_BLOCK_N = N // BLOCK_N
  NUM_BLOCK_K = K // BLOCK_K

  # Blocking N dimension (the RHS free dimension)
  for n in nl.affine_range(NUM_BLOCK_N):
    result_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                             nl.par_dim(TILE_M), TILE_N),
                            dtype=lhsT.dtype,
                            buffer=nl.sbuf)

    # Blocking K dimension (the contraction dimension)
    # Use `sequential_range` because we do not want the compiler to change this loop by, 
    # for example, vectorizing it
    for k in nl.sequential_range(NUM_BLOCK_K):
      # Loading tiles from rhs
      # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
      i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
      rhs_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                             dtype=rhs.dtype,
                             buffer=nl.sbuf)

      for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
        rhs_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
            rhs[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                BLOCK_N * n + i_rhs.x])

      # Blocking M dimension (the LHS free dimension)
      for m in nl.affine_range(NUM_BLOCK_M):
        # Loading tiles from lhsT
        i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
        lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                                dtype=lhsT.dtype,
                                buffer=nl.sbuf)
        for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
          lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
              lhsT[(TILES_IN_BLOCK_K * k + bk_l) * TILE_K + i_lhsT.p,
                   BLOCK_M * m + i_lhsT.x])

        # Do matmul with all tiles in the blocks
        i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
        i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]
        i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          for bm in nl.affine_range(TILES_IN_BLOCK_M):
            res_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

            for bk in nl.affine_range(TILES_IN_BLOCK_K):
              res_tile[...] += nisa.nc_matmul(
                  lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                  rhs_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x])

            # Accumulate on corresponding SBUF tile
            result_tiles[m, bm, bn, i_res_mm.p,
                         i_res_mm.x] += res_tile[i_res_mm.p, i_res_mm.x]

    # Copying the result from SBUF to HBM
    for m in nl.affine_range(NUM_BLOCK_M):
      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        i_res = nl.mgrid[0:TILE_M, 0:TILE_N]
        i_res_packed = nl.mgrid[0:TILE_M, 0:BLOCK_N]
        result_packed = nl.ndarray((TILE_M, BLOCK_N),
                                   dtype=result_tiles.dtype,
                                   buffer=nl.sbuf)

        # coalesce result tiles for better DMA performance
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          result_packed[i_res.p,
                        bn * TILE_N + i_res.x] = nl.copy(result_tiles[m, bm, bn,
                                                                      i_res.p,
                                                                      i_res.x])
        nl.store(result[(TILES_IN_BLOCK_M * m + bm) * TILE_M + i_res_packed.p,
                        BLOCK_N * n + i_res_packed.x],
                 value=result_packed[i_res_packed.p, i_res_packed.x])

  return result
```

## layernorm.html

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np
import math

@nki.jit
def nki_layernorm_kernel_v1(input_tensor, epsilon, gamma_vector, beta_vector):
  """Computes LayerNorm.
    Used nki.language APIs only.
  """
  output_tensor = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype,
                             buffer=nl.shared_hbm)

  # Ensure that the shapes of tensors match
  assert input_tensor.shape[1] == gamma_vector.shape[0] == beta_vector.shape[0]

  # Generate tile indices for loading/storing data
  i_p_io = nl.arange(nl.tile_size.pmax)[:, None]
  i_f_io = nl.arange(input_tensor.shape[1])[None, :]
  i_p_param = nl.arange(1)[:, None]

  # Number of rows in the input tensor
  num_rows = input_tensor.shape[0]

  # Load gamma and beta, which will be reused across rows/tiles of input_tensor
  gamma_sb = nl.load(gamma_vector.reshape((1, gamma_vector.shape[0]))[i_p_param, i_f_io])
  beta_sb = nl.load(beta_vector.reshape((1, beta_vector.shape[0]))[i_p_param, i_f_io])

  # Broadcast the gamma and beta to match the dimensions of the tiles
  gamma_sb_bcast = gamma_sb.broadcast_to((nl.tile_size.pmax, gamma_vector.shape[0]))
  beta_sb_bcast = beta_sb.broadcast_to((nl.tile_size.pmax, beta_vector.shape[0]))

  # Tile partition dimension of the input tensor by nl.tile_size.pmax
  for i in nl.affine_range(math.ceil(input_tensor.shape[0]/nl.tile_size.pmax)):
    # Load input tile
    input_sb = nl.load(input_tensor[i * nl.tile_size.pmax + i_p_io, i_f_io],
                       mask=(i * nl.tile_size.pmax + i_p_io < num_rows))

    # Compute mean and variance
    mean = nl.mean(input_sb, axis=1)
    # Trick to calculate var with mean: mean(x^2) - mean(x)^2
    var = nl.mean(nl.square(input_sb), axis=1) - mean * mean

    # Normalize the input by shifting with the mean 
    # and scaling with rsqrt of variance and epsilon
    shift_scale_tensor = (input_sb - mean) * nl.rsqrt(var + epsilon)
    
    # Scale the normalized tile using gamma and add beta
    output_sb = shift_scale_tensor * gamma_sb_bcast + beta_sb_bcast

    nl.store(output_tensor[i * nl.tile_size.pmax + i_p_io, i_f_io], value=output_sb,
             mask=(i * nl.tile_size.pmax + i_p_io < num_rows))

  return output_tensor
```

```python
@nki.jit
def nki_layernorm_kernel_v2(input_tensor, epsilon, gamma_vector, beta_vector):
  """Computes LayerNorm.
    Used nki.isa APIs to calculate mean/variance and perform shift/scale.
  """
  output_tensor = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype,
                             buffer=nl.shared_hbm)

  # Ensure that the shapes of tensors match
  assert input_tensor.shape[1] == gamma_vector.shape[0] == beta_vector.shape[0]

  # Generate tile indices for loading/storing data
  i_p_io = nl.arange(nl.tile_size.pmax)[:, None]
  i_f_io = nl.arange(input_tensor.shape[1])[None, :]
  i_p_param = nl.arange(1)[:, None]

  # Number of rows in the input tensor
  num_rows = input_tensor.shape[0]

  # Load gamma and beta, which will be reused across rows/tiles of input_tensor
  gamma_sb = nl.load(gamma_vector.reshape((1, gamma_vector.shape[0]))[i_p_param, i_f_io])
  beta_sb = nl.load(beta_vector.reshape((1, beta_vector.shape[0]))[i_p_param, i_f_io])

  # Broadcast the gamma and beta to match the dimensions of the tiles
  gamma_sb_bcast = gamma_sb.broadcast_to((nl.tile_size.pmax, gamma_vector.shape[0]))
  beta_sb_bcast = beta_sb.broadcast_to((nl.tile_size.pmax, beta_vector.shape[0]))

  # Tile partition dimension of the input tensor by nl.tile_size.pmax
  for i in nl.affine_range(math.ceil(input_tensor.shape[0]/nl.tile_size.pmax)):
    # Load input tile
    input_sb = nl.load(input_tensor[i * nl.tile_size.pmax + i_p_io, i_f_io],
                       mask=(i * nl.tile_size.pmax + i_p_io < num_rows))

    # Tile free dimension of the input tensor by nl.tile_size.bn_stats_fmax, 
    # as bn_stats has a free dimension size limit
    i_f_bn = nl.arange(nl.tile_size.bn_stats_fmax)[None, :]
    i_f_stats = nl.arange(6)[None, :]
    num_bn_stats = math.ceil(input_tensor.shape[1]/nl.tile_size.bn_stats_fmax)
    stats_results = nl.ndarray((nl.tile_size.pmax, 6*num_bn_stats), dtype=np.float32)
    for j in nl.affine_range(num_bn_stats):
      stats_results[i_p_io, j * 6 + i_f_stats] = nisa.bn_stats(
              input_sb[i_p_io, j * nl.tile_size.bn_stats_fmax + i_f_bn],
              mask=(j * nl.tile_size.bn_stats_fmax + i_f_bn < input_tensor.shape[1]),
              dtype=np.float32)
      
    # Aggregate bn_stats results to compute mean and var
    i_f_aggr = nl.arange(6*num_bn_stats)[None, :]
    mean_var = nisa.bn_aggr(stats_results[i_p_io, i_f_aggr])
    mean = mean_var[i_p_io, 0]
    var = mean_var[i_p_io, 1]

    # Get reciprocal of sqrt(var + epsilon)
    scale_var = nl.rsqrt(var + epsilon)

    # Putting the shift and scale together in one line to trigger two alu_op tensor_vector instruction
    shift_scale_tensor = nisa.tensor_scalar(data=input_sb, op0=np.subtract,
                                            operand0=mean,
                                            op1=np.multiply,
                                            operand1=scale_var)
    
    # Scale the normalized tile using gamma and add beta
    output_sb = shift_scale_tensor * gamma_sb_bcast + beta_sb_bcast

    nl.store(output_tensor[i * nl.tile_size.pmax + i_p_io, i_f_io], value=output_sb,
             mask=(i * nl.tile_size.pmax + i_p_io < num_rows))

  return output_tensor
```

## fused-self-attn.html

```python
import neuronx_distributed_training.nki as nki
import neuronx_distributed_training.nisa as nisa
import numpy as np

@nki.jit
def fused_self_attn_for_SD_small_head_size(q_ref, k_ref, v_ref, use_causal_mask=False,
                                           mixed_precision=True):
  """
  Fused self attention kernel for small head dimension Stable Diffusion workload, 
  simplified for this tutorial. 
  
  Computes softmax(QK^T)V. Decoder model can optionally include a causal mask 
  application. Does not include QKV projection, output projection, dropout, 
  residual connection, etc.

  This kernel is designed to be used for Stable Diffusion models where the 
  d_head is smaller or equal to 128. Assertion is thrown if `d_head` does
  not satisfy the requirement.

  IO tensor layouts:
   - q_ptr: shape   (seq_q, d_head)
   - k_ptr: shape   (seq_k, d_head)
   - v_ptr: shape   (seq_v, d_head)
   - out_ptr: shape (seq_q, d_head)
   - We use seq_q and seq_k and seq_v just for clarity, this kernel requires 
   seq_q == seq_k == seq_v

  IO tensor dtypes:
   - This kernel assumes all IO tensors have the same dtype
   - If mixed_precision is True, then all Tensor Engine operation will be performed in
   bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
   will be in the same type as the inputs.
  """
  # Use q_ref dtype as the intermediate tensor dtype
  # Assume all IO tensors have the same dtype
  kernel_dtype = q_ref.dtype
  pe_in_dt = nki.bfloat16 if mixed_precision else np.float32
  assert q_ref.dtype == k_ref.dtype == v_ref.dtype

  # Shape checking
  seqlen, d_head = q_ref.shape
  assert d_head <= 128, "Cannot use this kernel for d_head > 128"
  assert tuple(q_ref.shape) == (seqlen, d_head), 'Input shape mismatch!'
  assert tuple(k_ref.shape) == (seqlen, d_head), 'Input shape mismatch!'
  assert tuple(v_ref.shape) == (seqlen,d_head), \
  f'Input shape mismatch! Expected: {(seqlen, d_head)} Actual: {tuple(v_ref.shape)}'
  out_ref = nki.ndarray((seqlen, d_head), dtype=q_ref.dtype, buffer=nki.shared_hbm)

  # Softmax scaling factor, multiplied onto Q
  softmax_scale = 0.125

  q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128
  k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
  # No tiling on d_head dimension since the dimension of d_head fits in SB
  d_head_tile_size = d_head
  v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

  ###################################
  # Step 1. transpose(tensor_v)
  ###################################
  # Buffer for v matrix transposed
  # Pre-fetch and keep it in SBUF throughout different softmax tiles
  trans_v = nki.ndarray((nki.par_dim(v_seq_tile_size), v_seq_n_tiles, d_head), dtype=pe_in_dt)

  for i_k_seq_tile in nki.affine_range(k_seq_n_tiles):
    ip_v = nki.arange(v_seq_tile_size)[:, None]
    if_v = nki.arange(d_head_tile_size)[None, :]
    trans_v[ip_v, i_k_seq_tile, if_v] = nki.load(
      v_ref[i_k_seq_tile * k_seq_tile_size + ip_v, if_v],
      dtype=pe_in_dt)

  q_local = nki.ndarray((q_seq_n_tiles, nki.par_dim(d_head_tile_size), q_seq_tile_size), dtype=pe_in_dt)
  ip_q = nki.arange(d_head_tile_size)[:, None]
  if_q = nki.arange(q_seq_tile_size)[None, :]
  for i_q_seq_tile in nki.affine_range(q_seq_n_tiles):
    q_local[i_q_seq_tile, ip_q, if_q] = nki.load_transpose2d(
      q_ref[i_q_seq_tile * q_seq_tile_size + nki.arange(q_seq_tile_size)[:, None],
            nki.arange(d_head_tile_size)[None, :]
      ],
      dtype=pe_in_dt) * softmax_scale

  k_local = nki.ndarray((k_seq_n_tiles, nki.par_dim(d_head_tile_size), k_seq_tile_size), dtype=pe_in_dt)
  ip_k = nki.arange(d_head_tile_size)[:, None]
  if_k = nki.arange(k_seq_tile_size)[None, :]
  for i_k_seq_tile in nki.affine_range(k_seq_n_tiles):
    k_local[i_k_seq_tile, ip_k, if_k] = nki.load_transpose2d(
      k_ref[i_k_seq_tile * k_seq_tile_size + nki.arange(k_seq_tile_size)[:, None],
            nki.arange(d_head_tile_size)[None, :]],
      dtype=pe_in_dt)

  for i_q_seq_tile in nki.affine_range(q_seq_n_tiles):  # indent = 2
    # A SBUF buffer for an independent softmax tile
    qk_res_buf = nki.ndarray((nki.par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype)

    neg_max_res = nki.ndarray((nki.par_dim(q_seq_tile_size), k_seq_n_tiles), dtype=kernel_dtype)
    ip_max = nki.arange(q_seq_tile_size)[:, None]
    if_max = nki.arange(k_seq_n_tiles)[None, :]

    # Loop over RHS free of matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
    for i_k_seq_tile in nki.affine_range(k_seq_n_tiles):  # indent = 4

      # Since the K^T tile is the RHS, the q_seq_len dimension will be P in the result
      # PSUM buffer shape: [q_seq_tile_size P, k_seq_tile_size F]
      qk_psum = nki.zeros((nki.par_dim(q_seq_tile_size), k_seq_tile_size),
                         dtype=np.float32, buffer=nki.psum)

      # Tensor indices for accessing qk result in k_seq_tile_size
      ip_qk = nki.arange(q_seq_tile_size)[:, None]
      if_qk = nki.arange(k_seq_tile_size)[None, :]

      ##############################################################
      # Step 2. matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
      ##############################################################
      qk_psum[ip_qk, if_qk] += nisa.nc_matmul(moving=k_local[i_k_seq_tile, ip_k, if_k],
                                              stationary=q_local[i_q_seq_tile, ip_q, if_q])

      ###################################
      # Step 3. Apply optional causal mask
      ###################################
      if use_causal_mask:
        # Magic number nki.fp32.min to replace -inf similar to what neuronx-cc uses
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nisa.affine_select(
          pred=(i_q_seq_tile * q_seq_tile_size + ip_qk >= i_k_seq_tile * k_seq_tile_size + if_qk),
          on_true_tile=qk_psum[ip_qk, if_qk], on_false_value=nki.fp32.min, dtype=kernel_dtype)
      else:
        # Simply send psum result back to sbuf
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nki.copy(qk_psum[ip_qk, if_qk],
                                                                              dtype=kernel_dtype)

      ###################################
      # Step 4. Softmax
      ###################################
      neg_max_res[ip_max, i_k_seq_tile] = nisa.tensor_reduce(
        np.max, data=qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk],
        axis=(1,), dtype=kernel_dtype, negate=True)

    neg_max_res_final = nisa.tensor_reduce(
      np.min, data=neg_max_res[ip_max, if_max],
      axis=(1,), dtype=kernel_dtype, negate=False)

    ip_softmax = nki.arange(q_seq_tile_size)[:, None]
    if_softmax = nki.arange(seqlen)[None, :]
    ip_sum_res = nki.arange(q_seq_tile_size)[:, None]
    if_sum_res = nki.arange(d_head_tile_size)[None, :]

    softmax_res = nki.ndarray((nki.par_dim(q_seq_tile_size), seqlen), dtype=pe_in_dt)
    sum_divisor = nki.ndarray((nki.par_dim(q_seq_tile_size), d_head_tile_size), dtype=kernel_dtype)

    # Simply use a large tile of seq_len in size since this is a "blocking" instruction
    # Assuming the compiler will merge exp and reduce_add into a single instruction on ACT
    exp_res = nisa.activation(np.exp,
                              data=qk_res_buf[ip_softmax, if_softmax],
                              bias=neg_max_res_final, scale=1.0)

    sum_res = nisa.tensor_reduce(np.add, data=exp_res, axis=(1,),
                          dtype=kernel_dtype)
    softmax_res[ip_softmax, if_softmax] = nki.copy(exp_res, dtype=pe_in_dt)

    sum_reciprocal_broadcast = (1.0 / sum_res).broadcast_to((q_seq_tile_size, d_head_tile_size))
    sum_divisor[ip_sum_res, if_sum_res] = nki.copy(sum_reciprocal_broadcast, dtype=kernel_dtype)

    # Buffer for transposed softmax results (FP32 in PSUM)
    trans_softmax_res = nki.ndarray(
      (nki.par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
      dtype=pe_in_dt)

    # Result psum buffer has the hidden dim as P
    attn_res_psum = nki.zeros((nki.par_dim(d_head_tile_size), q_seq_tile_size),
                             dtype=np.float32, buffer=nki.psum)

    ip_scores_t = nki.arange(k_seq_tile_size)[:, None]
    if_scores_t = nki.arange(q_seq_tile_size)[None, :]
    # Loop over matmul_1 contraction
    for i_k_seq_tile in nki.affine_range(k_seq_n_tiles):
      ###################################
      # Step 5. transpose(softmax_res)
      ###################################
      ip_scores = nki.arange(q_seq_tile_size)[:, None]
      if_scores = nki.arange(k_seq_tile_size)[None, :]

      trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = nisa.nc_transpose(
        softmax_res[ip_scores, i_k_seq_tile * k_seq_tile_size + if_scores])

    ip_out = nki.arange(d_head_tile_size)[:, None]
    if_out = nki.arange(q_seq_tile_size)[None, :]
    for i_k_seq_tile in nki.affine_range(k_seq_n_tiles):
      ######################################################################
      # Step 6. matmul_1(stationary=trans_v, moving=trans_softmax_res, contract=seqlen_v=seqlen_k)
      ######################################################################
      ip_v_t = nki.arange(k_seq_tile_size)[:, None]
      if_v_t = nki.arange(d_head_tile_size)[None, :]
      attn_res_psum[ip_out, if_out] += \
        nisa.nc_matmul(moving=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                       stationary=trans_v[ip_v_t, i_k_seq_tile, if_v_t])

    attn_res_sbuf = nki.copy(attn_res_psum[ip_out, if_out], dtype=kernel_dtype)

    attn_res_div = attn_res_sbuf * nisa.nc_transpose(sum_divisor[ip_sum_res, if_sum_res])

    nki.store(
      out_ref[i_q_seq_tile * q_seq_tile_size + if_out, ip_out],
      value=attn_res_div)

  return out_ref
```

## spmd_tensor_addition.html

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit
def nki_tensor_add_kernel_(a_input, b_input):
  """NKI kernel to compute element-wise addition of two input tensors

  This kernel assumes strict input/output sizes can be uniformly tiled to [128,512]

  Args:
      a_input: a first input tensor
      b_input: a second input tensor

  Returns:
      c_output: an output tensor
  """
  # Create output tensor shared between all SPMD instances as result tensor
  c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)

  # Calculate tile offsets based on current 'program'
  offset_i_x = nl.program_id(0) * 128
  offset_i_y = nl.program_id(1) * 512

  # Generate tensor indices to index tensors a and b
  ix = offset_i_x + nl.arange(128)[:, None]
  iy = offset_i_y + nl.arange(512)[None, :]

  # Load input data from device memory (HBM) to on-chip memory (SBUF)
  # We refer to an indexed portion of a tensor as an intermediate tensor
  a_tile = nl.load(a_input[ix, iy])
  b_tile = nl.load(b_input[ix, iy])

  # compute a + b
  c_tile = a_tile + b_tile

  # store the addition results back to device memory (c_output)
  nl.store(c_output[ix, iy], value=c_tile)

  # Transfer the ownership of `c_output` to the caller
  return c_output


def nki_tensor_add(a_input, b_input):
  """NKI kernel caller to compute element-wise addition of two input tensors

  This kernel caller lifts tile-size restriction, by applying the kernel on tiles of the inputs/outputs

  Args:
      a_input: a first input tensor, of shape [N*128, M*512]
      b_input: a second input tensor, of shape [N*128, M*512]

  Returns:
      a tensor of shape [N*128, M*512], the result of a_input + b_input
  """

  # The SPMD launch grid denotes the number of kernel instances.
  # In this case, we use a 2D grid where the size of each invocation is 128x512
  grid_x = a_input.shape[0] // 128
  grid_y = a_input.shape[1] // 512

  return nki_tensor_add_kernel_[grid_x, grid_y](a_input, b_input)
```

## rmsnorm.html

```python
import math
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit
def nki_rmsnorm_kernel(a_tensor, g_tensor):
  # Calculate out_tensor = a_tensor/RMS(a_tensor) * g_tensor
  # Where RMS(a_tensor) = sqrt((1/N) * sum(a_tensor * a_tensor))
  # and N = a_tensor.shape[1]
  # Reduction (mean) is performed in the free (2nd) dimension
  out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                          buffer=nl.shared_hbm)

  # Make sure shapes match
  assert a_tensor.shape[1] == g_tensor.shape[0]

  # Generate tensor indices to index input tensor
  ix = nl.arange(128)[:, None]
  iw = nl.arange(1)[:, None]
  iy = nl.arange(a_tensor.shape[1])[None, :]

  num_rows = a_tensor.shape[0]

  # Load RMSNorm weight once, reused by rows/tiles of a_tensor
  g_tile = nl.load(g_tensor.reshape((1, g_tensor.shape[0]))[iw, iy])

  # Process 128 rows at a time due to 128-partition tile size limitation
  # Since we're not reducing across the first dimension
  # Tiles can be processed independently
  for i in nl.affine_range(math.ceil(a_tensor.shape[0]/128)):

    # Load input data from external memory to on-chip memory
    a_tile = nl.load(a_tensor[i * 128 + ix, iy],
                    mask=(i * 128 + ix < num_rows))

    # Compute element-wise square of a_tensor
    in_square = nl.square(a_tile)

    # Calculate sum of squared elements, along last dimension
    square_sum = nl.sum(in_square, axis=[1])

    # Scale and get a reciprocal
    mean = square_sum / a_tensor.shape[1]

    # Take square root of mean and then reciprocal with
    # rsqrt API (one ISA instruction)
    rms_reciprocal = nl.rsqrt(mean)

    # Scale the input tensor
    out_tile = nl.multiply(a_tile, rms_reciprocal)

    # Broadcast weight along first axis to match tensor shape
    # num_rows_active = min(num_rows - i * 128, 128)
    g_bcast = g_tile.broadcast_to((128, g_tensor.shape[0]))

    # Multiply with the RMSNorm weight
    out_tile[...] = nl.multiply(out_tile, g_bcast,
                           mask=(i * 128 + ix < num_rows))

    # store the addition results back to external memory (out_tensor)
    nl.store(out_tensor[i * 128 + ix, iy], value=out_tile,
            mask=(i * 128 + ix < num_rows))

  return out_tensor
```

```python
def torch_rmsnorm_kernel(a_tensor, g_tensor):
  # Square the tensor (element-wise)
  in_square = a_tensor.pow(2)
  # Calculate means in the free dimension
  mean = in_square.mean(dim=1, keepdim=True)
  # Scale by reciprocal of sqrt(mean)
  tensor = a_tensor * torch.rsqrt(mean)

  # Scale the output by the weight
  return tensor * g_tensor
```

```python
def jax_rms_norm(a_tensor, g_tensor):
  # Square the tensor (element-wise)
  in_square = jnp.square(a_tensor)
  # Calculate means in the free dimension
  mean = in_square.mean(axis=1, keepdims=True)
  # Scale by reciprocal of sqrt(mean)
  tensor = a_tensor * jnp.reciprocal(jnp.sqrt(mean))

  # Scale the output by the weight
  return tensor * g_tensor
```

## neuron_profile_for_nki.rst

```python
import os
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["NEURON_CC_FLAGS"]= " --disable-dge "
```

```python
# prof-kernel.py - Full code example (referenced but not shown in source)
# This example computes element-wise exponential of a 2D input tensor
# See: examples/prof-kernel.py
```

```python
# prof-kernel-profile.py - Using nki.profile API
# Full code example decorated with @nki.profile
# See: examples/prof-kernel-profile.py
```

**Note:** The source document references two complete code examples (`prof-kernel.py` and `prof-kernel-profile.py`) via `literalinclude` directives, but the actual code contents are not included in the provided source text. Only the environment variable setup code is explicitly shown. To extract the full API usage examples, the actual example files would need to be provided.

## getting_started.html

```python
from neuronxcc import nki
import neuronxcc.nki.language as nl


@nki.jit
def nki_tensor_add_kernel(a_input, b_input):
    """NKI kernel to compute element-wise addition of two input tensors"""

    # Check all input/output tensor shapes are the same for element-wise operation
    assert a_input.shape == b_input.shape

    # Check size of the first dimension does not exceed on-chip memory tile size limit,
    # so that we don't need to tile the input to keep this example simple
    assert a_input.shape[0] <= nl.tile_size.pmax

    # Load the inputs from device memory to on-chip memory
    a_tile = nl.load(a_input)
    b_tile = nl.load(b_input)

    # Specify the computation (in our case: a + b)
    c_tile = nl.add(a_tile, b_tile)

    # Create a HBM tensor as the kernel output
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)

    # Store the result to c_output from on-chip memory to device memory
    nl.store(c_output, value=c_tile)

    # Return kernel output as function output
    return c_output
```

## getting_started.rst

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
```

```python
@nki.jit
def nki_tensor_add_kernel(a_input, b_input):
    pass
```

```python
assert a_input.shape == b_input.shape
assert a_input.shape[0] <= nl.tile_size.pmax
```

```python
i_x = nl.arange(a_input.shape[0])[:, None]
i_y = nl.arange(a_input.shape[1])[None, :]

a_tile = nl.load(a_input[i_x, i_y])
b_tile = nl.load(b_input[i_x, i_y])
```

```python
c_tile = nl.add(a_tile, b_tile)
```

```python
c_output = nl.zeros(a_input.shape, dtype=a_input.dtype)
nl.store(c_output[i_x, i_y], c_tile)
return c_output
```

```python
import numpy as np

a = np.ones((4, 3), dtype=np.float16)
b = np.ones((4, 3), dtype=np.float16)
c = nki_tensor_add_kernel(a, b)
```

```python
import torch
import torch_neuronx

a = torch.ones((4, 3), dtype=torch.float16, device='xla:0')
b = torch.ones((4, 3), dtype=torch.float16, device='xla:0')
c = nki_tensor_add_kernel(a, b)
print(c)
```

```python
import jax
import jax.numpy as jnp
import libneuronxla

a = jnp.ones((4, 3), dtype=jnp.float16)
b = jnp.ones((4, 3), dtype=jnp.float16)
c = nki_tensor_add_kernel(a, b)
```

## nki_block_dimension_migration_guide.html

```python
import neuron.nki as nki
import neuron.nki.language as nl
from neuron.nki import ncc

# Example 1: Block dimension tensor allocation (original pattern)
@nki.jit
def exp_func(inp):
  output = nl.ndarray((4, 8, 128, 2, 512), dtype=float32,
    buffer=nl.shared_hbm)
  a = nl.ndarray((4, 8, nl.par_dim(128), 2, 512), dtype=float32, buffer=nl.sbuf)
  for i in range(4):
    for j in range(8):
      a[i, j] = nl.load(inp[i, j])
      a[i, j] = nl.exp(a[i, j])
      nl.store(output[i, j], value=result)
```

```python
import neuron.nki as nki
import neuron.nki.language as nl

# Example 2: SBUF migration - blocks alive simultaneously (move to free dimension)
@nki.jit
def sb_blocks(inp):
    res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
    add_buf = nl.ndarray(shape=(8, nl.par_dim(128), 512), dtype=inp.dtype, buffer=nl.sbuf)
    for i in range(8):
        add_buf[i] = nl.load(inp[i])
    for i in range(8):
        nl.store(res[i], add_buf[i])
    return res

# Migrated version
@nki.jit
def sb_blocks_migrated(inp):
    res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
    add_buf = nl.ndarray(shape=(128, 8, 512), dtype=inp.dtype, buffer=nl.sbuf)
    for i in range(8):
        add_buf[0:128, i, 0:512] = nl.load(inp[i])
    for i in range(8):
        nl.store(res[i], add_buf[0:128, i, 0:512])
    return res
```

```python
import neuron.nki as nki
import neuron.nki.language as nl

# Example 3: SBUF migration - blocks not simultaneously alive (hoist down)
@nki.jit
def sb_blocks(inp):
    res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
    add_buf = nl.ndarray(shape=(8, nl.par_dim(128), 512), dtype=inp.dtype, buffer=nl.sbuf)
    for i in range(8):
        add_buf[i] = nl.load(inp[i])
        nl.store(res[i], add_buf[i])
    return res

# Migrated version
@nki.jit
def sb_blocks_migrated(inp):
    res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
    for i in range(8):
        add_buf = nl.ndarray(shape=(128, 512), dtype=inp.dtype, buffer=nl.sbuf)
        add_buf[0:128, 0:512] = nl.load(inp[i])
        nl.store(res[i], add_buf[0:128, 0:512])
    return res
```

```python
import neuron.nki as nki
import neuron.nki.language as nl
from neuron.nki import ncc

# Example 4: Direct allocation with multi-buffering (original pattern)
def interleave_alloc_func(idx, pdim_size, fdim_size):
  idx, = idx
  start_partition = 0
  return (start_partition, (idx % 2) * fdim_size)

@nki.jit
def copy_func(inp):
  output = nl.ndarray((4, 128, 512), dtype=float32, buffer=nl.shared_hbm)
  a = nl.ndarray((4, nl.par_dim(128), 512), dtype=float32, buffer=ncc.sbuf.alloc(interleave_alloc_func))
  for i in range(4):
      a[i] = nl.load(inp[i])
      nl.store(output[i], value=a[i])
```

```python
import neuron.nki as nki
import neuron.nki.language as nl
from neuron.nki import ncc

# Example 5: Direct allocation with multi-buffering (migrated pattern)
def interleave_alloc_func(idx, pdim_size, fdim_size):
  assert idx == ()
  start_partition = 0
  return (start_partition, (idx % 2) * fdim_size)

@nki.compiler.skip_middle_end_transformations
@nki.jit
def exp_func(inp):
  output = nl.ndarray((4, 128, 512), dtype=nl.float32, buffer=nl.shared_hbm)
  a = nl.ndarray((128, 2, 512), dtype=nl.float32, buffer=ncc.sbuf.alloc(interleave_alloc_func))
  for i in range(4):
    a[0:128, i % 2, 0:512] = nl.load(inp[i])
    nl.store(output[i], value=a[0:128, i % 2, 0:512])
```

## transpose2d.html

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit
def tensor_transpose2D_kernel_(in_tensor, shape2D):
  """
  NKI kernel to reorder the elements on axis[1] of the input tensor.

  Every row of the input tensor is a flattened row-major 2D matrix.
  The shape2D argument defines the dimensions of the flattened matrices (#rows,#cols).
  Our goal in this kernel is to transpose these flattened 2D matrices, i.e. make them (#cols,#rows).

  Example:
      in_tensor = [a0,a1,a2,a3,b0,b1,b2,b3,c0,c1,c2,c3]
      shape2D = (3,4)
  this means that in_tensor has 3 rows and 4 columns, i.e. can be represented as:
      [a0,a1,a2,a3]
      [b0,b1,b2,b3]
      [c0,c1,c2,c3]
  after transpose, we expect to get:
      [a0,b0,c0]
      [a1,b1,c1]
      [a2,b2,c2]
      [a3,b3,c3]
  Thus, out_tensor is expected to be [a0,b0,c0,a1,b1,c1,a2,b2,c2,a3,b3,c3]

  Args:
    in_tensor: an input tensor
    shape2D: tuple representing the dimensions to be transposed: (#rows, #cols)
    out_tensor: an output (transposed) tensor
  """
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # Gather input shapes
  sz_p, _ = in_tensor.shape

  # Load input data from external memory to on-chip memory
  in_tile = nl.load(in_tensor)

  # Performing f1/f2 transpose
  # ==========================
  # The desired transpose pattern is provided as an input:
  sz_f1, sz_f2 = shape2D

  # We're going to need 3 indices to perform f1:f2 transpose.
  # - i_p0 is the parallel index
  # - i_f1 and i_f2 are both free-dim indices, and will be used to transpose between the f1/f2 axes
  i_p0 = nl.arange(sz_p)[:, None, None]
  i_f1 = nl.arange(sz_f1)[None, :, None]
  i_f2 = nl.arange(sz_f2)[None, None, :]

  # Perform the transposition via a SBUF-to-SBUF copy, with access-pattern manipulation
  # Note that we have 2D tensors and 3 indices, since we need to represent a 2D access pattern *per partition*
  # RHS traverses an F1 x F2 matrix in a row major manner
  # LHS traverses an F2 x F1 (new) matrix in a row major manner
  out_tile = nl.ndarray(shape=(sz_p, sz_f2*sz_f1), dtype=out_tensor.dtype)
  out_tile[i_p0, i_f2*sz_f1+i_f1] = nl.copy(in_tile[i_p0, i_f1*sz_f2+i_f2])

  # Finally, we store out_tile to external memory
  nl.store(out_tensor, value=out_tile)

  return out_tensor
```

## average_pool2d.html

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor

@nki.jit
def tensor_avgpool_kernel(in_tensor, pool_size):
  """NKI kernel to compute a 2D avg-pool operation

  Args:
      in_tensor: an input tensor, of shape C x H x W
      pool_size: an integer representing a (square) pool-window size

  Return:
      out_tensor: the resulting output tensor, of shape C x (H/pool_size) x (W/pool_size)
  """

  # Get input/output dimensions
  sz_cin, sz_hin, sz_win = in_tensor.shape
  sz_hout = sz_hin // pool_size
  sz_wout = sz_win // pool_size
  # Create output tensor shared between all SPMD instances as result tensor
  out_tensor = nl.ndarray((sz_cin, sz_hout, sz_wout), dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  # Set relevant sizes
  sz_p = sz_cin
  sz_pool = pool_size

  # Generate pool index patterns (requires two extra dimensions, for the pool window)
  i0, i1, i2, i3, i4 = nl.mgrid[0:sz_p, 0:sz_hin//sz_pool, 0:sz_pool, 0:sz_win//sz_pool, 0:sz_pool]

  # Load input data from external memory to on-chip memory
  in_tile: tensor[sz_p, sz_hin, sz_win] = nl.load(in_tensor)

  # Perform the pooling operation:
  # We use numpy's advanced indexing, in order to extend in_tile to 5D, and then reduce-average two dimension.
  # axis[0] is the index for p_dim, and thus doesn't participate in the reduction operation.
  # axis[1] and axis[2] together index the rows, with axis[2] responsible for inner strides
  # (i.e. inside a pooling window), and axis[1] responsible for the outer strides. As such, we reduce over axis[2].
  # Similarly, axis[3] and axis[4] together index the columns, and we thus reduce over axis[4].
  out_tile : tensor[sz_p, sz_hout, sz_wout] = nl.sum(in_tile[i0, sz_pool*i1+i2, sz_pool*i3+i4],
                                                     axis=[2,4]) / (pool_size*pool_size)

  # Store the results back to hbm
  nl.store(out_tensor, value=out_tile)

  # Transfer the ownership of `out_tensor` to the caller
  return out_tensor
```

```python
import jax.numpy as jnp

# Reference JAX implementation
def jax_average_pool_2D(in_tensor, pool_size):
  c, h_in, w_in = in_tensor.shape
  reshaped = in_tensor.reshape(c, h_in // pool_size, pool_size, w_in // pool_size, pool_size)
  return jnp.nanmean(reshaped, axis=(2, 4))
```

## nki.isa.nc_matmul.html

```python
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

##################################################################
# Example 1:
# multiply matrix a of shape (128, 128) and matrix b of shape (128, 512)
# to get matrix c in PSUM of shape (128, 512)
##################################################################
a_mgrid = nl.mgrid[0:128, 0:128]
b_mgrid = nl.mgrid[0:128, 0:512]
c_mgrid = nl.mgrid[0:128, 0:512]

a = nl.load(a_tensor[a_mgrid.p, a_mgrid.x])
b = nl.load(b_tensor[b_mgrid.p, b_mgrid.x])

c_psum = nisa.nc_matmul(a[a_mgrid.p, a_mgrid.x], b[b_mgrid.p, b_mgrid.x])

nl.store(c_tensor[c_mgrid.p, c_mgrid.x], c_psum)

##################################################################
# Example 2:
# multiply matrix d of shape (256, 128) and matrix e of shape (256, 512)
# to get matrix f in PSUM of shape (128, 512) using psum accumulation
##################################################################
d_mgrid = nl.mgrid[0:128, 0:128]
e_mgrid = nl.mgrid[0:128, 0:512]
f_mgrid = nl.mgrid[0:128, 0:512]

f_psum = nl.zeros((128, 512), nl.float32, buffer=nl.psum)

for i_contract in nl.affine_range(2):
  d = nl.load(d_tensor[i_contract * 128 + d_mgrid.p, d_mgrid.x])
  e = nl.load(e_tensor[i_contract * 128 + e_mgrid.p, e_mgrid.x])
  f_psum += nisa.nc_matmul(d[d_mgrid.p, d_mgrid.x], e[e_mgrid.p, e_mgrid.x])
    
nl.store(f_tensor[f_mgrid.p, f_mgrid.x], f_psum)

##################################################################
# Example 3:
# perform batched matrix multiplication on matrix g of shape (16, 64, 64) 
# and matrix h of shape (16, 64, 512) to get matrix i of (16, 64, 512) 
# using Tensor Engine PE tiling mode. 
##################################################################
g_mgrid = nl.mgrid[0:64, 0:64]
h_mgrid = nl.mgrid[0:64, 0:512]
i_mgrid = nl.mgrid[0:64, 0:512]

for i in nl.affine_range(4):
  for j in nl.affine_range(4):
    g = nl.load(g_tensor[i * 4 + j, g_mgrid.p, g_mgrid.x])
    h = nl.load(h_tensor[i * 4 + j, h_mgrid.p, h_mgrid.x])
    i_psum = nisa.nc_matmul(g, h, tile_position=((i % 2) * 64, (j % 2) * 64), tile_size=(64, 64))
    nl.store(i_tensor[i * 4 + j, i_mgrid.p, i_mgrid.x], i_psum)
```

## nki.baremetal.html

```python
from neuronxcc.nki import baremetal
import neuronxcc.nki.language as nl
import numpy as np

@baremetal(save_neff_name='file.neff', save_trace_name='profile.ntff')
def nki_tensor_tensor_add(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

  a = nl.load(a_tensor)
  b = nl.load(b_tensor)

  c = a + b

  nl.store(c_tensor, c)

  return c_tensor
```

## nki.language.add.html

```python
import neuronxcc.nki.language as nl

a = nl.load(a_tensor[0:128, 0:512])
b = nl.load(b_tensor[0:128, 0:512])
# add a and b element-wise and store in c[128, 512]
c = nl.add(a, b)
nl.store(c_tensor[0:128, 0:512], c)
```

```python
import neuronxcc.nki.language as nl

a = nl.load(a_tensor[0:128, 0:512])
b = 2.2
# add constant b to each element in a
c = nl.add(a, b)
nl.store(c_tensor[0:128, 0:512], c)
```

```python
import neuronxcc.nki.language as nl

a = nl.load(a_tensor[0:128, 0:512])
b = nl.load(b_tensor[0:128, 0:1])
# broadcast on free dimension -- [128, 1] is broadcasted to [128, 512]
c = nl.add(a, b)
nl.store(c_tensor[0:128, 0:512], c)
```

```python
import neuronxcc.nki.language as nl

a = nl.load(a_tensor[0:128, 0:512])
b = nl.load(b_tensor[0:1, 0:512])
# broadcast on partition dimension -- [1, 512] is broadcasted to [128, 512]
c = nl.add(a, b)
nl.store(c_tensor[0:128, 0:512], c)
```

```python
import neuronxcc.nki.language as nl

a = nl.load(a_tensor[0:128, 0:512])
b = nl.load(b_tensor[0:1, 0:1])
# broadcast on both dimensions -- [1, 1] is broadcasted to [128, 512]
c = nl.add(a, b)
nl.store(c_tensor[0:128, 0:512], c)
```

```python
import neuronxcc.nki.language as nl

a = nl.load(a_tensor[0:128, 0:1])
b = nl.load(b_tensor[0:1, 0:512])
# broadcast on each dimensions -- [128, 1] and [1, 512] are broadcasted to [128, 512]
c = nl.add(a, b)
nl.store(c_tensor[0:128, 0:512], c)
```

## mm-nisa-spmd.py

```python
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc import nki

@nki.jit
def matmul_128x128x512_spmd(A_T, B):
    """NKI kernel to compute a 128x128x512 matrix multiplication operation
    Use SPMD program IDs to index into the full A and B input tensor to get tiles for 128x128x512 matrix multiplication.
    
    Args:
        A_T: an input tensor of shape [K=128,M=512], a left hand side argument of the matrix multiplication,
        B: an input tensor of shape [K=128,N=1024], a right hand side argument of the matrix multiplication
    
    result: the resulting output tensor of shape [M=512,N=1024]
    """
    K, N = A_T.shape
    K_, M = B.shape
    assert K == K_
    
    # Create output tensor shared between all SPMD instances as result tensor
    result = nl.ndarray((N, M), dtype=A_T.dtype, buffer=nl.shared_hbm)
    
    # Defining starting indexes for input A.T and B
    i_A_T_col = nl.program_id(0) * 128
    i_B_col = nl.program_id(1) * 512
    
    # Loading the inputs (HBM->SBUF)
    A_T_tile = nl.load(A_T[0:128, i_A_T_col:i_A_T_col+128])
    B_tile = nl.load(B[0:128, i_B_col:i_B_col+512])
    
    # Perform the matrix-multiplication
    result_psum = nisa.nc_matmul(A_T_tile, B_tile)
    
    # Copy the result from PSUM back to SBUF, and cast to expected output data-type
    result_sbuf = nl.copy(result_psum, dtype=result.dtype)
    
    # Store back into result tile with the correct SPMD offsets
    nl.store(result[i_A_T_col:i_A_T_col+128, i_B_col:i_B_col+512], value=result_sbuf)
    
    return result
```

## getting_started_baremetal.py

```python
from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_tensor_add_kernel(a_input, b_input):
    """NKI kernel to compute element-wise addition of two input tensors"""
    # Check all input/output tensor shapes are the same for element-wise operation
    assert a_input.shape == b_input.shape
    # Check size of the first dimension does not exceed on-chip memory tile size limit,
    # so that we don't need to tile the input to keep this example simple
    assert a_input.shape[0] <= nl.tile_size.pmax
    
    # Load the inputs from device memory to on-chip memory
    a_tile = nl.load(a_input)
    b_tile = nl.load(b_input)
    
    # Specify the computation (in our case: a + b)
    c_tile = nl.add(a_tile, b_tile)
    
    # Create a HBM tensor as the kernel output
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)
    # Store the result to c_output from on-chip memory to device memory
    nl.store(c_output, value=c_tile)
    # Return kernel output as function output
    return c_output
```

## getting_started_torch.py

```python
from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_tensor_add_kernel(a_input, b_input):
    """NKI kernel to compute element-wise addition of two input tensors"""
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)
    
    # Check all input/output tensor shapes are the same for element-wise operation
    assert a_input.shape == b_input.shape
    
    # Check size of the first dimension does not exceed on-chip memory tile size limit
    assert a_input.shape[0] <= nl.tile_size.pmax
    
    # Load the inputs from device memory to on-chip memory
    a_tile = nl.load(a_input)
    b_tile = nl.load(b_input)
    
    # Specify the computation (in our case: a + b)
    c_tile = nl.add(a_tile, b_tile)
    
    # Store the result to c_output from on-chip memory to device memory
    nl.store(c_output, value=c_tile)
    
    return c_output
```

## nki.jit.html

```python
from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_tensor_tensor_add(a_tensor, b_tensor):
  c_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)

  a = nl.load(a_tensor)
  b = nl.load(b_tensor)

  c = a + b

  nl.store(c_tensor, c)

  return c_tensor
```

## getting_started_jax.py

```python
from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_tensor_add_kernel(a_input, b_input):
    """NKI kernel to compute element-wise addition of two input tensors"""
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)
    
    # Check all input/output tensor shapes are the same for element-wise operation
    assert a_input.shape == b_input.shape
    
    # Check size of the first dimension does not exceed on-chip memory tile size limit
    assert a_input.shape[0] <= nl.tile_size.pmax
    
    # Load the inputs from device memory to on-chip memory
    a_tile = nl.load(a_input)
    b_tile = nl.load(b_input)
    
    # Specify the computation (in our case: a + b)
    c_tile = nl.add(a_tile, b_tile)
    
    # Store the result to c_output from on-chip memory to device memory
    nl.store(c_output, value=c_tile)
    
    return c_output
```