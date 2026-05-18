## fused_mamba.html

SUMMARY: This document demonstrates how to implement a fused Mamba SSM layer using NKI, covering key optimization techniques including associative scan operations, tensor tiling, loop reordering for data reuse, and seq_len tiling to minimize SBUF spilling and maximize compute engine utilization.

```python
import nki
import nki.language as nl
import nki.isa as nisa
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

    assert channels % 128 == 0

    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize

    for i_batch in nl.affine_range(batch_size):
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize

            scanC_accum = nl.zeros((channel_psize, seq_len), dtype=delta.dtype)

            for i_state in nl.affine_range(state_size):

                delta_slice = delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len]
                delta_i = nl.ndarray(delta_slice.shape, dtype=delta_slice.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=delta_i, src=delta_slice)
                A_slice = A[channel_start:channel_start+channel_psize, i_state:i_state+1]
                A_i = nl.ndarray(A_slice.shape, dtype=A_slice.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=A_i, src=A_slice)

                deltaA = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
                nisa.activation(dst=deltaA, op=nl.exp, data=delta_i, scale=A_i)

                u_slice = u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len]
                u_i = nl.ndarray(u_slice.shape, dtype=u_slice.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=u_i, src=u_slice)
                B_slice = B[i_batch, i_state:i_state+1, 0:seq_len]
                B_i = nl.ndarray(B_slice.shape, dtype=B_slice.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=B_i, src=B_slice)

                deltaU = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=deltaU, data1=delta_i, data2=u_i, op=nl.multiply)
                B_i_bcast = nl.broadcast_to(B_i, (channel_psize, seq_len))
                deltaBu = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=deltaBu, data1=deltaU, data2=B_i_bcast, op=nl.multiply)

                scan_res = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor_scan(dst=scan_res, data0=deltaA, data1=deltaBu, initial=0.0,
                        op0=nl.multiply, op1=nl.add)

                C_slice = C[i_batch, i_state:i_state+1, 0:seq_len]
                C_i = nl.ndarray(C_slice.shape, dtype=C_slice.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=C_i, src=C_slice)

                C_i_bcast = nl.broadcast_to(C_i, (channel_psize, seq_len))
                scanC = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=scanC, data1=scan_res, data2=C_i_bcast, op=nl.multiply)

                nisa.tensor_tensor(dst=scanC_accum, data1=scanC_accum, data2=scanC, op=nl.add)

            nisa.dma_copy(dst=output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
                    src=scanC_accum)

    return output
```

```python
@nki.jit
def mamba_v2(delta, u, A, B, C):
    """Computes the SSM operation in the Mamba model with loop reordering optimization.

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

    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize

    for i_batch in nl.affine_range(batch_size):

        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize

            scanC_accum = nl.zeros((channel_psize, seq_len), dtype=delta.dtype)

            delta_slice = delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len]
            delta_i = nl.ndarray(delta_slice.shape, dtype=delta_slice.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=delta_i, src=delta_slice)
            u_slice = u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len]
            u_i = nl.ndarray(u_slice.shape, dtype=u_slice.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=u_i, src=u_slice)

            for i_state in nl.affine_range(state_size):
                A_slice = A[channel_start:channel_start+channel_psize, i_state:i_state+1]
                A_i = nl.ndarray(A_slice.shape, dtype=A_slice.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=A_i, src=A_slice)

                deltaA = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
                nisa.activation(dst=deltaA, op=nl.exp, data=delta_i, scale=A_i)

                B_slice = B[i_batch, i_state:i_state+1, 0:seq_len]
                B_i = nl.ndarray(B_slice.shape, dtype=B_slice.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=B_i, src=B_slice)

                deltaU = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=deltaU, data1=delta_i, data2=u_i, op=nl.multiply)
                B_i_bcast = nl.broadcast_to(B_i, (channel_psize, seq_len))
                deltaBu = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=deltaBu, data1=deltaU, data2=B_i_bcast, op=nl.multiply)

                scan_res = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor_scan(dst=scan_res, data0=deltaA, data1=deltaBu, initial=0.0,
                        op0=nl.multiply, op1=nl.add)

                C_slice = C[i_batch, i_state:i_state+1, 0:seq_len]
                C_i = nl.ndarray(C_slice.shape, dtype=C_slice.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=C_i, src=C_slice)

                C_i_bcast = nl.broadcast_to(C_i, (channel_psize, seq_len))
                scanC = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=scanC, data1=scan_res, data2=C_i_bcast, op=nl.multiply)

                nisa.tensor_tensor(dst=scanC_accum, data1=scanC_accum, data2=scanC, op=nl.add)

            nisa.dma_copy(dst=output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
                    src=scanC_accum[0:channel_psize, 0:seq_len])

    return output
```

## kernel-optimization.html

SUMMARY: This document covers NKI kernel optimization techniques for AWS Neuron, demonstrating how to write and progressively optimize matrix multiplication kernels using the NKI API, profiling tools, and architectural understanding of the NeuronEngine to transform memory-bound kernels into compute-bound ones.

```python
import nki
import nki.language as nl
import nki.isa as nisa
import os

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

@nki.jit
def matrix_multiply_kernel(lhsT, rhs):
  """NKI kernel to compute a matrix multiplication operation on a single tile

  Args:
    lhsT: an input tensor of shape [K,M], where both K and M are, at most,
      128.  It is the left-hand-side argument of the matrix multiplication,
      delivered transposed for optimal performance.
    rhs: an input tensor of shape [K,N], where K is, at most, 128, and N
      is, at most, 512.  It is the right-hand-side argument of the matrix
      multiplication.
  Returns:
    result: the resulting output tensor of shape [M,N]
  """
  K, M = lhsT.shape
  K_, N = rhs.shape

  assert K == K_, \
    f"Contraction demention {K} does not match {K_}, did you remember to transpose?"

  assert K <= nl.tile_size.pmax, \
    f"Expected partition dimension in lhsT ({K}) to be less than {nl.tile_size.pmax}"
  assert M <= nl.tile_size.gemm_stationary_fmax, \
    f"Expected free dimension in lhsT ({M}) to be less than " \
    f"{nl.tile_size.gemm_stationary_fmax}"
  assert N <= nl.tile_size.gemm_moving_fmax, \
    f"Expected free dimension in rhs ({N}) to be less than " \
    f"{nl.tile_size.gemm_moving_fmax}"

  lhsT_tile = nl.ndarray(shape=lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
  rhs_tile = nl.ndarray(shape=rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)

  nisa.dma_copy(dst=lhsT_tile, src=lhsT)
  nisa.dma_copy(dst=rhs_tile, src=rhs)

  result_tile = nl.ndarray(shape=(M, N), dtype=nl.float32, buffer=nl.psum)
  nisa.nc_matmul(dst=result_tile, stationary=lhsT_tile, moving=rhs_tile)

  result_tmp = nl.ndarray(shape=result_tile.shape,
                          dtype=result_tile.dtype,
                          buffer=nl.sbuf)
  nisa.tensor_copy(dst=result_tmp, src=result_tile)

  result = nl.ndarray(shape=result_tmp.shape,
                      dtype=result_tmp.dtype,
                      buffer=nl.hbm)
  nisa.dma_copy(dst=result, src=result_tmp)

  return result
```

```python
import nki
import nki.language as nl
import nki.isa as nisa
import os

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

@nki.jit
def matrix_multiply_kernel(lhsT, rhs):
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

  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  assert M % TILE_M == 0, \
    f"Expected M, {M}, to be a multiple of stationary free-dimension max, {TILE_M}"
  assert N % TILE_N == 0, \
    f"Expected N, {N}, to be a multiple of moving free-dimension max, {TILE_N}"
  assert K % TILE_K == 0, \
    f"Expected K, {K}, to be a multiple of the partition dimension max, {TILE_K}"

  result = nl.ndarray(shape=(M, N), dtype=lhsT.dtype, buffer=nl.hbm)

  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      result_tile = nl.ndarray(shape=(TILE_M, TILE_N),
                           dtype=nl.float32,
                           buffer=nl.psum)

      for k in nl.affine_range(K // TILE_K):
        lhsT_tile = nl.ndarray(shape=(TILE_K, TILE_M),
                           dtype=lhsT.dtype,
                           buffer=nl.sbuf)
        rhs_tile = nl.ndarray(shape=(TILE_K, TILE_N),
                          dtype=rhs.dtype,
                          buffer=nl.sbuf)

        nisa.dma_copy(dst=lhsT_tile,
                  src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                           m * TILE_M:(m + 1) * TILE_M])
        nisa.dma_copy(dst=rhs_tile,
                  src=rhs[k * TILE_K:(k + 1) * TILE_K,
                          n * TILE_N:(n + 1) * TILE_N])

        nisa.nc_matmul(dst=result_tile, stationary=lhsT_tile, moving=rhs_tile)

      result_tmp = nl.ndarray(shape=(TILE_M, TILE_N),
                          dtype=nl.float32,
                          buffer=nl.sbuf)
      nisa.tensor_copy(dst=result_tmp, src=result_tile)

      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M,
                           n * TILE_N:(n + 1) * TILE_N],
                src=result_tmp)

  return result
```

```python
import nki
import nki.language as nl
import nki.isa as nisa
import os

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

@nki.jit
def matrix_multiply_kernel(lhsT, rhs):
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
  result = nl.ndarray(shape=(M, N), dtype=nl.float32, buffer=nl.hbm)

  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  assert M % TILE_M == 0, \
    f"Expected M, {M}, to be a multiple of stationary free-dimension max, {TILE_M}"
  assert N % TILE_N == 0, \
    f"Expected N, {N}, to be a multiple of moving free-dimension max, {TILE_N}"
  assert K % TILE_K == 0, \
    f"Expected K, {K}, to be a multiple of the partition dimension max, {TILE_K}"

  for m in nl.affine_range(M // TILE_M):
    lhsT_tiles = []
    for k in nl.affine_range(K // TILE_K):
      lhsT_tile = nl.ndarray(shape=(TILE_K, TILE_M),
                           dtype=lhsT.dtype,
                           buffer=nl.sbuf)
      nisa.dma_copy(dst=lhsT_tile,
                  src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                         m * TILE_M:(m + 1) * TILE_M])
      lhsT_tiles.append(lhsT_tile)

    for n in nl.affine_range(N // TILE_N):
      rhs_tiles = []
      for k in nl.affine_range(K // TILE_K):
        rhs_tile = nl.ndarray(shape=(TILE_K, TILE_N),
                          dtype=rhs.dtype,
                          buffer=nl.sbuf)
        nisa.dma_copy(dst=rhs_tile,
                  src=rhs[k * TILE_K:(k + 1) * TILE_K,
                          n * TILE_N:(n + 1) * TILE_N])
        rhs_tiles.append(rhs_tile)

      result_tile = nl.ndarray(shape=(TILE_M, TILE_N),
                           dtype=nl.float32,
                           buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
        nisa.nc_matmul(dst=result_tile,
                   stationary=lhsT_tiles[k],
                   moving=rhs_tiles[k])

      result_tmp = nl.ndarray(shape=(TILE_M, TILE_N),
                          dtype=nl.float32,
                          buffer=nl.sbuf)
      nisa.tensor_copy(dst=result_tmp, src=result_tile)

      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M,
                           n * TILE_N:(n + 1) * TILE_N],
                src=result_tmp)

  return result
```

```python
import nki
import nki.language as nl
import nki.isa as nisa
import os

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

@nki.jit
def matrix_multiply_kernel(lhsT, rhs):
  """NKI kernel to compute a matrix multiplication operation while blocking the
     free dimensions of the LHS and RHS to improve memory access pattern.

  Args:
      lhsT: an input tensor of shape [K,M], where both K and M are multiples for
        1.    It is the left-hand-side argument of the matrix multiplication,
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

  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  TILES_IN_BLOCK_M = 2
  TILES_IN_BLOCK_N = 2

  BLOCK_M = TILE_M * TILES_IN_BLOCK_M  # 256
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N  # 1024

  assert M % BLOCK_M == 0, f"Expected M ({M}) to be divisible by BLOCK_M ({BLOCK_M})"
  assert N % BLOCK_N == 0, f"Expected N ({N}) to be divisible by BLOCK_N ({BLOCK_N})"

  result = nl.ndarray(shape=(M, N), dtype=lhsT.dtype, buffer=nl.hbm)

  for m in nl.affine_range(M // BLOCK_M):
    lhsT_tiles = []
    for bm in nl.affine_range(TILES_IN_BLOCK_M):
      lhsT_tiles_internal = []
      for k in nl.affine_range(K // TILE_K):
        lhsT_tile = nl.ndarray(shape=(TILE_K, TILE_M),
                               dtype=lhsT.dtype,
                               buffer=nl.sbuf)
        nisa.dma_copy(dst=lhsT_tile,
                src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                     (m * TILES_IN_BLOCK_M + bm) *
                     TILE_M:((m * TILES_IN_BLOCK_M + bm) + 1) *
                     TILE_M])
        lhsT_tiles_internal.append(lhsT_tile)
      lhsT_tiles.append(lhsT_tiles_internal)

    for n in nl.affine_range(N // BLOCK_N):
      rhs_tiles = []
      for bn in nl.affine_range(TILES_IN_BLOCK_N):
        rhs_tiles_internal = []
        for k in nl.affine_range(K // TILE_K):
          rhs_tile = nl.ndarray(shape=(TILE_K, TILE_N),
                                dtype=rhs.dtype,
                                buffer=nl.sbuf)
          nisa.dma_copy(dst=rhs_tile,
                src=rhs[k * TILE_K:(k + 1) * TILE_K,
                    (n * TILES_IN_BLOCK_N + bn) *
                    TILE_N:((n * TILES_IN_BLOCK_N + bn) + 1) *
                    TILE_N])
          rhs_tiles_internal.append(rhs_tile)
        rhs_tiles.append(rhs_tiles_internal)

      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          result_tile = nl.ndarray(shape=(TILE_M, TILE_N),
                                   dtype=nl.float32,
                                   buffer=nl.psum)
          for k in nl.affine_range(K // TILE_K):
            nisa.nc_matmul(dst=result_tile,
                           stationary=lhsT_tiles[bm][k],
                           moving=rhs_tiles[bn][k])

          result_tmp = nl.ndarray(shape=result_tile.shape,
                                  dtype=result.dtype,
                                  buffer=nl.sbuf)
          nisa.tensor_copy(dst=result_tmp, src=result_tile)

          nisa.dma_copy(dst=result[(m * TILES_IN_BLOCK_M + bm) *
                                   TILE_M:((m * TILES_IN_BLOCK_M + bm) + 1) *
                                   TILE_M,
                                   (n * TILES_IN_BLOCK_N + bn) *
                                   TILE_N:((n * TILES_IN_BLOCK_N + bn) + 1) *
                                   TILE_N],
                        src=result_tmp)

  return result
```

```python
import nki
import nki.language as nl
import nki.isa as nisa
import os

os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

@nki.jit
def matrix_multiply_kernel(
    lhsT,
    rhs,
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

  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  BLOCK_M = TILE_M * TILES_IN_BLOCK_M
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N
  BLOCK_K = TILE_K * TILES_IN_BLOCK_K

  assert M % BLOCK_M == 0, \
    f"Expected M {M} to be divisble by {BLOCK_M} when there are {TILES_IN_BLOCK_M}"
  assert N % BLOCK_N == 0, \
    f"Expected N {N} to be divisble by {BLOCK_N} when there are {TILES_IN_BLOCK_N}"
  assert K % BLOCK_K == 0, \
    f"Expected K {K} to be divisble by {BLOCK_K} when there are {TILES_IN_BLOCK_K}"

  result = nl.ndarray(shape=(M,N), dtype=nl.float32, buffer=nl.hbm)

  NUM_BLOCK_M = M // BLOCK_M
  NUM_BLOCK_N = N // BLOCK_N
  NUM_BLOCK_K = K // BLOCK_K

  for n in nl.affine_range(NUM_BLOCK_N):
    result_tmps = []
    for m_idx in range(NUM_BLOCK_M):
      block_m = []
      for bm_idx in range(TILES_IN_BLOCK_M):
        block_n = []
        for bn_idx in range(TILES_IN_BLOCK_N):
          tile = nl.ndarray(shape=(TILE_M, TILE_N),
                            dtype=lhsT.dtype,
                            buffer=nl.sbuf)
          nisa.memset(dst=tile, value=0.0)
          block_n.append(tile)
        block_m.append(block_n)
      result_tmps.append(block_m)

    for k in nl.sequential_range(NUM_BLOCK_K):
      rhs_tiles = []
      for bk_r in range(TILES_IN_BLOCK_K):
        rhs_tile = nl.ndarray(shape=(TILE_K, BLOCK_N),
                              dtype=rhs.dtype,
                              buffer=nl.sbuf)
        nisa.dma_copy(dst=rhs_tile[0:TILE_K, 0:BLOCK_N],
                      src=rhs[(TILES_IN_BLOCK_K * k + bk_r) *
                              TILE_K:(TILES_IN_BLOCK_K * k + bk_r + 1) * TILE_K,
                              BLOCK_N * n:BLOCK_N * (n + 1)])
        rhs_tiles.append(rhs_tile)

      for m in nl.affine_range(NUM_BLOCK_M):
        lhsT_tiles = []
        for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
          lhsT_tile = nl.ndarray(shape=(TILE_K, BLOCK_M),
                                 dtype=lhsT.dtype,
                                 buffer=nl.sbuf)
          nisa.dma_copy(
            dst=lhsT_tile[0:TILE_K, 0:BLOCK_M],
            src=lhsT[(TILES_IN_BLOCK_K * k + bk_l) *
                 TILE_K:(TILES_IN_BLOCK_K * k + bk_l + 1) * TILE_K,
                 BLOCK_M * m:BLOCK_M * (m + 1)])
          lhsT_tiles.append(lhsT_tile)

        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          for bm in nl.affine_range(TILES_IN_BLOCK_M):
            result_tile = nl.ndarray(shape=(TILE_M, TILE_N),
                                     dtype=nl.float32,
                                     buffer=nl.psum)
            for bk in nl.affine_range(TILES_IN_BLOCK_K):
              nisa.nc_matmul(
                dst=result_tile,
                stationary=lhsT_tiles[bk][0:TILE_K, bm * TILE_M:(bm + 1) * TILE_M],
                moving=rhs_tiles[bk][0:TILE_K, bn * TILE_N:(bn + 1) * TILE_N]
              )
            nisa.tensor_tensor(dst=result_tmps[m][bm][bn],
                               data1=result_tmps[m][bm][bn],
                               data2=result_tile,
                               op=nl.add)

    for m in nl.affine_range(NUM_BLOCK_M):
      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        result_packed = nl.ndarray(shape=(TILE_M, BLOCK_N),
                                   dtype=nl.float32,
                                   buffer=nl.sbuf)
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          nisa.tensor_copy(
            dst=result_packed[0:TILE_M, bn * TILE_N:(bn + 1) * TILE_N],
            src=result_tmps[m][bm][bn][0:TILE_M, 0:TILE_N])

        nisa.dma_copy(dst=result[(TILES_IN_BLOCK_M * m + bm) *
                                 TILE_M:(TILES_IN_BLOCK_M * m + bm + 1) * TILE_M,
                                 BLOCK_N * n:BLOCK_N * (n + 1)],
                      src=result_packed[0:TILE_M, 0:BLOCK_N])

  return result
```

## mxfp-matmul.html

SUMMARY: This document covers MXFP4/8 matrix multiplication on AWS Neuron using NKI, demonstrating microscaling quantization, layout requirements, and best practices for writing MX kernels with APIs like quantize_mx, nc_matmul_mx, and strided access patterns.

```python
@nki.jit
def kernel_offline_quantized_mx_matmul(stationary_mx_data, stationary_mx_scale, moving_mx_data, moving_mx_scale, mx_dtype):    
  
  MAX_TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  MAX_TILE_K = nl.tile_size.pmax  # 128
  MAX_TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # View the input data as _x4 mx_dtype. This is done using an access pattern, specifying the target dtype and a simple
  # linear pattern.
  stationary_mx_data_hbm_x4 = stationary_mx_data.ap(dtype=mx_dtype, pattern=[[MAX_TILE_M,MAX_TILE_K],[1,MAX_TILE_M]], offset=0)
  moving_mx_data_hbm_x4 = moving_mx_data.ap(dtype=mx_dtype, pattern=[[MAX_TILE_N,MAX_TILE_K],[1,MAX_TILE_N]], offset=0)

  # Check that the input tiles are max-sized. This is merely for simplicity of the example but
  # smaller shapes are also supported.
  assert stationary_mx_data_hbm_x4.shape == (MAX_TILE_K, MAX_TILE_M)
  assert moving_mx_data_hbm_x4.shape == (MAX_TILE_K, MAX_TILE_N)

  # Load inputs directly from HBM to SBUF. Data is assumed to already have the 
  # layout required by MX. Scales are assumed to be contiguous in HBM therefore we use
  # load_scales_scattered() to spread them across SBUF partition-dim quadrants, as is required
  # by Matmul-MX.

  stationary_mx_data_sbuf_x4 = nl.ndarray(stationary_mx_data_hbm_x4.shape, dtype=mx_dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=stationary_mx_data_sbuf_x4, src=stationary_mx_data_hbm_x4)
  stationary_mx_scale_sbuf = load_scales_scattered(stationary_mx_data_sbuf_x4, stationary_mx_scale)

  # Load moving
  moving_mx_data_sbuf_x4 = nl.ndarray(moving_mx_data_hbm_x4.shape, dtype=mx_dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=moving_mx_data_sbuf_x4, src=moving_mx_data_hbm_x4)
  moving_mx_scale_sbuf = load_scales_scattered(moving_mx_data_sbuf_x4, moving_mx_scale)
  
  # Allocate a tile in PSUM. This could also be float32.
  result_psum = nl.ndarray((MAX_TILE_M, MAX_TILE_N), dtype=nl.bfloat16, buffer=nl.psum)

  # Matmul-MX
  nisa.nc_matmul_mx(
    dst=result_psum,
    stationary=stationary_mx_data_sbuf_x4,
    moving=moving_mx_data_sbuf_x4,
    stationary_scale=stationary_mx_scale_sbuf,
    moving_scale=moving_mx_scale_sbuf
  )

  # Copy the PSUM result back to SBUF
  result_sbuf = nl.ndarray(result_psum.shape, dtype=nl.bfloat16, buffer=nl.sbuf)
  nisa.tensor_copy(dst=result_sbuf, src=result_psum)  

  # Store to HBM
  result_hbm = nl.ndarray(result_psum.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)  
  nisa.dma_copy(dst=result_hbm, src=result_sbuf)
  
  return result_hbm
```

```python
def allocate_mx_tiles(shape_unquantized, mx_dtype, alloc_scale: bool = True):
  assert len(shape_unquantized) == 2, f"shape_unquantized must have exactly 2 dimensions, got {len(shape_unquantized)}"
  
  P, F = shape_unquantized
  
  # Allocate data tile
  # Quantize-MX shrinks the free-dim by 4x because it packs 4 elements into 1.
  mx_data_sbuf = nl.ndarray((P, F//4), dtype=mx_dtype, buffer=nl.sbuf)

  if not alloc_scale:
      return mx_data_sbuf, None
  
  # Allocate scale tile
  # Nominally the scale tile is sized (P//8, F//4) given that the scaling
  # group shape is [8P, 4F]. But when P > 32, the scales must be placed in the
  # partition-dim quadrant from which the corresponding scaling group originated 
  # hence we must allocate the full P.
  if P <= 32: # Can store all scales in first p-dim quadrant.
    mx_scale_sbuf = nl.ndarray((P//8, F//4), dtype=nl.uint8, buffer=nl.sbuf)
  else: # Must oversize and spread across quadrants.
    mx_scale_sbuf = nl.ndarray((P, F//4), dtype=nl.uint8, buffer=nl.sbuf)
  
  return mx_data_sbuf, mx_scale_sbuf
```

```python
@nki.jit
def kernel_on_device_quantize_matmul_mx(stationary_mx_data, stationary_mx_scale, moving_data_bf16, stationary_mx_dtype, moving_mx_dtype):

  assert moving_mx_dtype != nl.float4_e2m1fn_x4, "FP4 not supported by Quantize-MX"

  MAX_TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  MAX_TILE_K = nl.tile_size.pmax  # 128
  MAX_TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # View the input MX data as _x4 mx_dtype. This is done using an access pattern, specifying the target dtype and a simple
  # linear pattern.
  stationary_mx_data_hbm_x4 = stationary_mx_data.ap(dtype=stationary_mx_dtype, pattern=[[MAX_TILE_M,MAX_TILE_K],[1,MAX_TILE_M]], offset=0)

  # Check that the input tiles are max-sized. This is merely for simplicity of the example but
  # smaller shapes are also supported.
  assert stationary_mx_data_hbm_x4.shape == (MAX_TILE_K, MAX_TILE_M)
  # Note the factor of 4 on the N free-dim. This is unquantized data whose free-dim will be packed and
  # reduced by a factor of 4 during quantize_mx.
  assert moving_data_bf16.shape == (MAX_TILE_K, MAX_TILE_N*4)

  # Load stationary MX.
  stationary_mx_data_sbuf_x4 = nl.ndarray(stationary_mx_data_hbm_x4.shape, dtype=stationary_mx_dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=stationary_mx_data_sbuf_x4, src=stationary_mx_data_hbm_x4)
  stationary_mx_scale_sbuf = load_scales_scattered(stationary_mx_data_sbuf_x4, stationary_mx_scale)
  
  # Load moving BF16
  moving_bf16_sbuf = nl.ndarray(moving_data_bf16.shape, dtype=moving_data_bf16.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=moving_bf16_sbuf, src=moving_data_bf16)

  # Allocate quantized moving tiles
  moving_mx_data_sbuf_x4, moving_mx_scale_sbuf = allocate_mx_tiles(moving_data_bf16.shape, moving_mx_dtype)  

  # Quantize-MX. Scales will automatically be spread across partition-dim quadrants.
  nisa.quantize_mx(dst=moving_mx_data_sbuf_x4,
                  src=moving_bf16_sbuf,
                  dst_scale=moving_mx_scale_sbuf)  

  # Allocate a tile in PSUM
  result_psum = nl.ndarray((MAX_TILE_M, MAX_TILE_N), dtype=nl.bfloat16, buffer=nl.psum)

  # Matmul-MX
  nisa.nc_matmul_mx(
    dst=result_psum,
    stationary=stationary_mx_data_sbuf_x4,
    moving=moving_mx_data_sbuf_x4,
    stationary_scale=stationary_mx_scale_sbuf,
    moving_scale=moving_mx_scale_sbuf
  )  

  # Copy the PSUM result back to SBUF
  result_sbuf = nl.ndarray(result_psum.shape, dtype=nl.bfloat16, buffer=nl.sbuf)
  nisa.tensor_copy(dst=result_sbuf, src=result_psum)  

  # Store to HBM
  result_hbm = nl.ndarray(result_psum.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)  
  nisa.dma_copy(dst=result_hbm, src=result_sbuf)

  return result_hbm
```

```python
def copy_data_strided(stationary_hbm, moving_hbm, use_tensor_copy: bool = True):  
    
  # The HBM tensors have nominal shape [P,F]. Reshape into [4, P//4, F]. 
  # In other words, we divide the contraction axis into 4 "P" tiles since we'll eventually
  # need to read data from each tile and pack them together on SBUF.
  
  # These dimensions reflect the shape of each "P" tile.
  P_st = stationary_hbm.shape[0] // 4
  F_st = stationary_hbm.shape[1]
  P_mv = moving_hbm.shape[0] // 4
  F_mv = moving_hbm.shape[1]
  
  stationary_hbm_reshape = stationary_hbm.reshape((4, P_st, F_st))
  moving_hbm_reshape = moving_hbm.reshape((4, P_mv, F_mv))

  # Allocate SBUF tensors to store the strided result.
  # The shape is [P//4, F, 4] where the [P,F] is the shape of the unquantized input tensor.
  # In other words, we view the free-dim as having F_st/F_mv groups of 4 elements.
  # Taking 3D views of both the HBM and SBUF tensors allows for cleaner indexing.
  stationary_sbuf_strided = nl.ndarray((P_st, F_st, 4), dtype=stationary_hbm.dtype, buffer=nl.sbuf)
  moving_sbuf_strided = nl.ndarray((P_mv, F_mv, 4), dtype=moving_hbm.dtype, buffer=nl.sbuf)    

  # Perform a TensorCopy to achieve the required layout.
  if (use_tensor_copy):

    # First load from HBM -> SBUF. Take "P" tiles from HBM and write them
    # contiguously (adjacent to each other) into the SBUF free-dim. 
    # This load is not the focus of this example so its details are encapsulated in load_tensor_helper().
    # The SBUF shapes will be stationary_sbuf [P_st, 4, F_st], moving_sbuf [P_mv, 4, F_mv]
    stationary_sbuf, moving_sbuf = load_tensor_helper(stationary_hbm_reshape, moving_hbm_reshape)

    # Perform SBUF-to-SBUF TensorCopy to shuffle the data into the required MX layout.
    # Here are some tips on how to read this access pattern (AP).
    # .ap(pattern) = tuple of [step_size, count], right-most is the inner (fastest changing) dimension of the access pattern (AP).
    # The dst (*_strided) has no AP specified, meaning it is linearly written to.
    # To understand the src AP it's useful to refer to the SBUF Layout diagram in load_tensor_helper().
    # We read 1 element, then step F elements to the next tile, 4 times total. In other words, we gather a group
    # of 4 elements (one from each tile).
    # Then step 1 element and repeat the above F times to read an entire row of SBUF.
    # Then step to the next row of SBUF and repeat the above for all P rows of SBUF.
    # Note, this example is shown as a strided-read but it could be re-written as a strided-write, though it will be slower.
    # Secondly, the source tile can be in PSUM (i.e. the result of a prior matmul).
  
    nisa.tensor_copy(src=stationary_sbuf.ap(pattern=[[4*F_st, P_st], [1, F_st], [F_st, 4]], offset=0), dst=stationary_sbuf_strided)
    nisa.tensor_copy(src=moving_sbuf.ap(pattern=[[4*F_mv, P_mv], [1, F_mv], [F_mv, 4]], offset=0), dst=moving_sbuf_strided)

  # Perform a strided DMA to achieve the required layout.
  else:

    # Similar to TensorCopy, the we linearly write to stationary_sbuf_strided.
    # When reading from *_hbm_reshape, we read one element from each tile.
    # Then step 1 element and repeat the above F times, thereby reading one full row of HBM.
    # Then step to the next row of HBM and repeat the above P times.

    nisa.dma_copy(src=stationary_hbm_reshape.ap(pattern=[[F_st, P_st], [1, F_st], [P_st*F_st, 4]], offset=0),
                  dst=stationary_sbuf_strided)
    nisa.dma_copy(src=moving_hbm_reshape.ap(pattern=[[F_mv, P_mv], [1, F_mv], [P_mv*F_mv, 4]], offset=0),
                  dst=moving_sbuf_strided)

  # Return as 2D.
  return stationary_sbuf_strided.reshape((P_st, F_st*4)), moving_sbuf_strided.reshape((P_mv, F_mv*4))
```

```python
@nki.jit
def kernel_copy_strided_quantize_matmul_mx_packed_scale(stationary_hbm, moving_hbm, mx_dtype, use_tensor_copy: bool = True):
  
  assert mx_dtype != nl.float4_e2m1fn_x4, "FP4 not supported by Quantize-MX"
 
  MAX_TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  MAX_TILE_K = nl.tile_size.pmax  # 128
  MAX_TILE_N = nl.tile_size.gemm_moving_fmax  # 512  

  # Ensure input tensors are in HBM.
  assert stationary_hbm.buffer == moving_hbm.buffer == nl.hbm

  # Sanity check the shapes. We expect contraction dimension of the unquantized tile to be 4x.
  assert stationary_hbm.shape == (MAX_TILE_K*4, MAX_TILE_M)
  assert moving_hbm.shape == (MAX_TILE_K*4, MAX_TILE_N)

  # Use strided access patterns to achieve required MX layout.
  # Returned shape is [P//4, F*4] where [P,F] is the input shape.
  stationary_sbuf_strided, moving_sbuf_strided = copy_data_strided(stationary_hbm, moving_hbm, use_tensor_copy)

  # Allocate quantized stationary/moving tiles.
  # Unlike the example kernel_copy_strided_quantize_matmul_mx, we do not allocate scale tiles here.
  stationary_mx_data_sbuf, _  = allocate_mx_tiles(stationary_sbuf_strided.shape, mx_dtype, alloc_scale=False)
  moving_mx_data_sbuf, _ = allocate_mx_tiles(moving_sbuf_strided.shape, mx_dtype, alloc_scale=False)

  # Allocate a single tile into which we will pack scale values from BOTH quantize_mx calls.
  #
  # quantize_mx requires that the input tile's free dimension contains exactly 4x as many 
  # elements as the scale tile. We will use this tile for both quantize_mx calls, so its 
  # free dimension needs to be able to hold the larger of the two input tiles, hence MAX_TILE_N.
  packed_mx_scale_sbuf = nl.ndarray((MAX_TILE_K, MAX_TILE_N), dtype=nl.uint8, buffer=nl.sbuf)

  # Each scaling group consists of 32 elements, with 8 partitions x 4 elements per partition.
  # Therefore, for each 32-partition SBUF quadrant, we get only 32 // 8 = 4 partitions' worth of scale factors.
  # This leaves 28 partitions unused. quantize_mx lets us use some of this space by storing other tensors'
  # scale factors at an offset.

  # In this example, we use tensor slicing to store:
  # - stationary's scale values at offset 0 in each quadrant (i.e., partitions 0:4, 32:36, 64:68, 96:100)
  # - moving's scale values at offset 4 in each quadrant (i.e., partitions 4:8, 36:40, 68:72, 100:104)

  # moving's scale values will be written to partitions 0:4 in each quadrant.
  # Additionally, we restrict the free dimension size to match stationary's shape.
  stationary_mx_scale_sbuf = packed_mx_scale_sbuf[0:, :MAX_TILE_M]

  # moving's scale values will be written to partitions 4:8 in each quadrant.
  # We don't restrict the size of the free dimension; it already matches moving's shape.
  moving_mx_scale_sbuf = packed_mx_scale_sbuf[4:, :]

  # Quantize-MX. Scales will automatically be spread across partition-dim quadrants.
  nisa.quantize_mx(dst=stationary_mx_data_sbuf,
                  src=stationary_sbuf_strided,
                  dst_scale=stationary_mx_scale_sbuf)

  nisa.quantize_mx(dst=moving_mx_data_sbuf,
                  src=moving_sbuf_strided,
                  dst_scale=moving_mx_scale_sbuf)
  
  # Allocate a tile in PSUM
  result_psum = nl.ndarray((MAX_TILE_M, MAX_TILE_N), dtype=nl.bfloat16, buffer=nl.psum)

  # Matmul-MX
  nisa.nc_matmul_mx(
    dst=result_psum,
    stationary=stationary_mx_data_sbuf,
    moving=moving_mx_data_sbuf,
    stationary_scale=stationary_mx_scale_sbuf,
    moving_scale=moving_mx_scale_sbuf
  )

  # Copy the PSUM result back to SBUF
  result_sbuf = nl.ndarray(result_psum.shape, dtype=nl.bfloat16, buffer=nl.sbuf)
  nisa.tensor_copy(dst=result_sbuf, src=result_psum)  

  # Store to HBM
  result_hbm = nl.ndarray(result_psum.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)  
  nisa.dma_copy(dst=result_hbm, src=result_sbuf)

  return result_hbm
```

## matrix_multiplication.html

SUMMARY: This document demonstrates NKI kernel programming for matrix multiplication on AWS Trainium, showing progressive optimization techniques including tiling, load hoisting, dimension blocking, and memory management using SBUF, PSUM, and HBM buffers.

```python
import neuron.nki as nki
import neuron.nki.isa as nisa
import neuron.nki.language as nl

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
  # Verify that the lhsT and rhs are the expected sizes.
  K, M = lhsT.shape
  K_, N = rhs.shape

  # Check that the contraction dimension matches and all dimensions
  #are what were expected.
  assert K == K_, \
    f"Expected contraction dimension to match on both lhsT ({K}) and rhs ({K})"
  assert K == 128, f"Expected contraction dimension to be 128, but got {K}"
  assert M == 64, f"Expected lhsT matrix to have dimension M of 64, but got {M}"
  assert N == 512, f"Expected rhs matrix to have dimension N of 512, but got {N}"

  # Create a tensor to write the result into (not initialized)
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Creating a tensor in SBUF to load the inputs into (not initialized)
  lhs_tile = nl.ndarray(lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
  rhs_tile = nl.ndarray(rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)

  # Loading the inputs (HBM->SBUF)
  # Note: here we take Tile dtype definition into account,
  # which forces P-dim as the left most index
  nisa.dma_copy(dst=lhs_tile, src=lhsT)
  nisa.dma_copy(dst=rhs_tile, src=rhs)

  # Create a tensor in PSUM to accumulate the result in (uninitialized)
  result_psum = nl.ndarray(result.shape, dtype=nl.float32, buffer=nl.psum)

  # Perform the matrix-multiplication
  # Note: A NKI matmul instruction always writes to PSUM in float32 data-type
  nisa.nc_matmul(result_psum, lhs_tile, rhs_tile)

  # Create a tensor in SBUF and copy the result from PSUM back to SBUF, 
  # and cast to expected output data-type
  result_sbuf = nl.ndarray(result_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
  nisa.tensor_copy(dst=result_sbuf, src=result_psum)

  # The result of [64,128] x [128,512] matrix multiplication has a shape of [64, 512].
  # This dictates which indices to use to address the result tile.
  nisa.dma_copy(dst=result, src=result_sbuf)

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

  # Verify that the lhsT and rhs have the same contraction dimension.
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  # Lookup the device matrix multiply dimensions.
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # Verify that the input matrices are a multiple of the tile dimensions.
  assert M % TILE_M == 0, \
    f"Expected M, {M}, to be a multiple of stationary free-dimension max, {TILE_M}"
  assert N % TILE_N == 0, \
    f"Expected N, {N}, to be a multiple of moving free-dimension max, {TILE_N}"
  assert K % TILE_K == 0, \
    f"Expected K, {K}, to be a multiple of the partition dimension max, {TILE_K}"

  # Create a space for the result in HBM (not initialized)
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Use affine_range to loop over tiles
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      # Allocate a tensor in PSUM
      res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

      for k in nl.affine_range(K // TILE_K):
        # Declare the tiles on SBUF
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

        # Load tiles from lhsT and rhs
        nisa.dma_copy(dst=lhsT_tile,
                      src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                               m * TILE_M:(m + 1) * TILE_M])
        nisa.dma_copy(dst=rhs_tile, 
                      src=rhs[k * TILE_K:(k + 1) * TILE_K,
                              n * TILE_N:(n + 1) * TILE_N])

        # Accumulate partial-sums into PSUM
        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

      # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
      nisa.tensor_copy(dst=res_sb, src=res_psum)

      # Copy the result from SBUF to HBM.
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M,
                               n * TILE_N:(n + 1) * TILE_N],
                    src=res_sb)

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

  # Verify that the lhsT and rhs are the expected sizes.
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  # Lookup the device matrix multiply dimensions.
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # Verify that the input matrices are a multiple of the tile dimensions.
  assert M % TILE_M == 0, \
    f"Expected M, {M}, to be a multiple of stationary free-dimension max, {TILE_M}"
  assert N % TILE_N == 0, \
    f"Expected N, {N}, to be a multiple of moving free-dimension max, {TILE_N}"
  assert K % TILE_K == 0, \
    f"Expected K, {K}, to be a multiple of the partition dimension max, {TILE_K}"

  # Create a space for the result in HBM (not initialized)
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Use affine_range to loop over tiles
  for m in nl.affine_range(M // TILE_M):
    # Load a whole column tiles from lhsT (with K * TILE_M numbers)
    # This corresponds to the whole row in the original lhs
    lhsT_tiles = []
    for k in nl.affine_range(K // TILE_K):
      # Allocate space in SBUF for the tile (uninitialized)
      lhsT_tile = nl.ndarray(shape=(TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
      # Copy the tile from HBM to SBUF
      nisa.dma_copy(dst=lhsT_tile, 
                    src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                             m * TILE_M:(m + 1) * TILE_M])
      # Append the tile to the list of tiles.
      lhsT_tiles.append(lhsT_tile)

    for n in nl.affine_range(N // TILE_N):
      # Load a whole column tiles from rhs (with K * TILE_N numbers)
      rhs_tiles = []
      for k in nl.affine_range(K // TILE_K):
        # Allocate space in SBUF for the tile (uninitialized)
        rhs_tile = nl.ndarray(shape=(TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
        # Copy the tile from HBM to SBUF
        nisa.dma_copy(dst=rhs_tile,
                      src=rhs[k * TILE_K:(k + 1) * TILE_K,
                              n * TILE_N:(n + 1) * TILE_N])
        # Append the tile to the list of tiles.
        rhs_tiles.append(rhs_tile)

      # Allocate a tile in PSUM for the result (uninitialized)
      res_psum = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
        # Accumulate partial-sums into PSUM
        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tiles[k], moving=rhs_tiles[k])

      # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
      nisa.tensor_copy(dst=res_sb, src=res_psum)

      # Copy the result from SBUF to HBM.
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M,
                               n * TILE_N:(n + 1) * TILE_N],
                    src=res_sb)

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

  # Verify that the lhsT and rhs have the same contraction dimension.
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  # Lookup the device matrix multiply dimensions.
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # Configuring the blocking size for the free dimensions
  TILES_IN_BLOCK_M = 2
  TILES_IN_BLOCK_N = 2

  BLOCK_M = TILE_M * TILES_IN_BLOCK_M  # 256
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N  # 1024

  # the size has to be multiple of block size
  assert M % BLOCK_M == 0
  assert N % BLOCK_N == 0

  # Create a space for the result in HBM (not initialized)
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Loop over blocks over the M dimension
  for m in nl.affine_range(M // BLOCK_M):
    # Load TILES_IN_BLOCK_M columns tiles by TILES_K rows from lhsT
    lhsT_tiles = []
    for bm in nl.affine_range(TILES_IN_BLOCK_M):
      # Inner tile array.
      lhsT_tiles_internal = []
      for k in nl.affine_range(K // TILE_K):
        # Allocate space in SBUF for the tile (uninitialized)
        lhsT_tile = nl.ndarray(shape=(TILE_K, TILE_M),
                               dtype=lhsT.dtype,
                               buffer=nl.sbuf)
        # Copy the tile from HBM to SBUF
        nisa.dma_copy(dst=lhsT_tile,
                      src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                               (m * TILES_IN_BLOCK_M + bm) *
                               TILE_M:((m * TILES_IN_BLOCK_M + bm) + 1) *
                               TILE_M])
        # Append the tile to the inner list of tiles.
        lhsT_tiles_internal.append(lhsT_tile)
      # Append the inner list of tiles into the outer list of tiles.
      lhsT_tiles.append(lhsT_tiles_internal)

    for n in nl.affine_range(N // BLOCK_N):
      # Load TILES_IN_BLOCK_N columns from rhs by TILES_K rows from rhs
      rhs_tiles = []
      for bn in nl.affine_range(TILES_IN_BLOCK_N):
        # Inner tile array.
        rhs_tiles_internal = []
        for k in nl.affine_range(K // TILE_K):
          # Allocate space in SBUF for the tile (uninitialized)
          rhs_tile = nl.ndarray(shape=(TILE_K, TILE_N),
                                dtype=rhs.dtype,
                                buffer=nl.sbuf)
          # Copy the tile from HBM to SBUF
          nisa.dma_copy(dst=rhs_tile,
                        src=rhs[k * TILE_K:(k + 1) * TILE_K,
                                (n * TILES_IN_BLOCK_N + bn) *
                                TILE_N:((n * TILES_IN_BLOCK_N + bn) + 1) *
                                TILE_N])
          # Append the tile to the inner list of tiles.
          rhs_tiles_internal.append(rhs_tile)
        # Append the inner list of tiles into the outer list of tiles.
        rhs_tiles.append(rhs_tiles_internal)

      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          # Allocate a tensor in PSUM
          result_tile = nl.ndarray(shape=(TILE_M, TILE_N),
                                   dtype=nl.float32,
                                   buffer=nl.psum)
          for k in nl.affine_range(K // TILE_K):
            # Accumulate partial-sums into PSUM
            nisa.nc_matmul(dst=result_tile,
                           stationary=lhsT_tiles[bm][k],
                           moving=rhs_tiles[bn][k])
  
          # Copy the result from PSUM back to SBUF, and cast to expected
          # output data-type
          result_tmp = nl.ndarray(shape=result_tile.shape,
                                  dtype=result.dtype,
                                  buffer=nl.sbuf)
          nisa.tensor_copy(dst=result_tmp, src=result_tile)

          # Copy the result from SBUF to HBM.
          nisa.dma_copy(dst=result[(m * TILES_IN_BLOCK_M + bm) *
                                   TILE_M:((m * TILES_IN_BLOCK_M + bm) + 1) *
                                   TILE_M,
                                   (n * TILES_IN_BLOCK_N + bn) *
                                   TILE_N:((n * TILES_IN_BLOCK_N + bn) + 1) *
                                   TILE_N],
                        src=result_tmp)

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

  # Verify that the lhsT and rhs have the same contraction dimension.
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  # Lookup the device matrix multiply dimensions.
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # Compute the block dimensions.
  BLOCK_M = TILE_M * TILES_IN_BLOCK_M
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N
  BLOCK_K = TILE_K * TILES_IN_BLOCK_K

  # Verify the size is a multiple of block size
  assert M % BLOCK_M == 0, \
    f"Expected M {M} to be divisble by {BLOCK_M} when there are {TILES_IN_BLOCK_M}"
  assert N % BLOCK_N == 0, \
    f"Expected N {N} to be divisble by {BLOCK_N} when there are {TILES_IN_BLOCK_N}"
  assert K % BLOCK_K == 0, \
    f"Expected K {K} to be divisble by {BLOCK_K} when there are {TILES_IN_BLOCK_K}"

  # Create a space for the result in HBM (not initialized)
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Compute the number of blocks in each dimension
  NUM_BLOCK_M = M // BLOCK_M
  NUM_BLOCK_N = N // BLOCK_N
  NUM_BLOCK_K = K // BLOCK_K

  # Blocking N dimension (the RHS free dimension)
  for n in nl.affine_range(NUM_BLOCK_N):
    # Create the initial result tiles in SBUF and initialize each tile to
    # 0.0, since the final results will be accumulated here. Results in 3-d array.
    result_tmps = []
    for m_idx in range(NUM_BLOCK_M):
      block_m = []
      for bm_idx in range(TILES_IN_BLOCK_M):
        block_n = []
        for bn_idx in range(TILES_IN_BLOCK_N):
          # Create the result tile (uninitialized)
          tile = nl.ndarray(shape=(TILE_M, TILE_N), dtype=lhsT.dtype, buffer=nl.sbuf)
          # Initialize the tile 0.0
          nisa.memset(dst=tile, value=0.0)
          # Append the tile to block_n array.
          block_n.append(tile)
        # Append block_n array to block_m array.
        block_m.append(block_n)
      # Append block_m array into result_tmps.
      result_tmps.append(block_m)

    # Blocking K dimension (the contraction dimension)
    # Use `sequential_range` because we do not want the compiler
    # to change this loop by, for example, vectorizing it
    for k in nl.sequential_range(NUM_BLOCK_K):
      # Loading tiles from rhs
      # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
      rhs_tiles = []
      for bk_r in range(TILES_IN_BLOCK_K):
        # Allocate rhs_tile tensor, TILE_K x BLOCK_N
        rhs_tile = nl.ndarray(shape=(TILE_K, BLOCK_N),
                              dtype=rhs.dtype,
                              buffer=nl.sbuf)
        # Copy block tile from rhs, to rhs_tile.
        nisa.dma_copy(dst=rhs_tile[0:TILE_K, 0:BLOCK_N],
                      src=rhs[(TILES_IN_BLOCK_K * k + bk_r) *
                              TILE_K:(TILES_IN_BLOCK_K * k + bk_r + 1) * TILE_K,
                              BLOCK_N * n:BLOCK_N * (n + 1)])
        # Append rhs_tile to rhs_tiles.
        rhs_tiles.append(rhs_tile)


      # Blocking M dimension (the LHS free dimension)
      for m in nl.affine_range(NUM_BLOCK_M):
        # Loading tiles from lhsT
        lhsT_tiles = []
        for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
          # Allocate lhsT_tile in SBUF (uninitialized)
          lhsT_tile = nl.ndarray(shape=(TILE_K, BLOCK_M),
                                 dtype=lhsT.dtype,
                                 buffer=nl.sbuf)
          # Copy block tile from lhsT to lhsT_tile
          nisa.dma_copy(dst=lhsT_tile[0:TILE_K, 0:BLOCK_M],
                        src=lhsT[(TILES_IN_BLOCK_K * k + bk_l) *
                                 TILE_K:(TILES_IN_BLOCK_K * k + bk_l + 1) * TILE_K,
                                 BLOCK_M * m:BLOCK_M * (m + 1)])
          # Append to list of lhsT tiles.
          lhsT_tiles.append(lhsT_tile)

        # Do matmul with all tiles in the blocks
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          for bm in nl.affine_range(TILES_IN_BLOCK_M):
            # Allocate result_tile in PSUM (uninitialized)
            result_tile = nl.ndarray(shape=(TILE_M, TILE_N),
                                     dtype=nl.float32,
                                     buffer=nl.psum)
            for bk in nl.affine_range(TILES_IN_BLOCK_K):
              # Perform matrix multiply on a tile.
              nisa.nc_matmul(
                dst=result_tile,
                stationary=lhsT_tiles[bk][0:TILE_K, bm * TILE_M:(bm + 1) * TILE_M],
                moving=rhs_tiles[bk][0:TILE_K, bn * TILE_N:(bn + 1) * TILE_N]
              )
            # Accumulate the result into the result_tmps tile.
            nisa.tensor_tensor(dst=result_tmps[m][bm][bn],
                               data1=result_tmps[m][bm][bn],
                               data2=result_tile,
                               op=nl.add)

    # Copying the result from SBUF to HBM
    for m in nl.affine_range(NUM_BLOCK_M):
      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        # coalesce result tiles for better DMA performance
        result_packed = nl.ndarray(shape=(TILE_M, BLOCK_N),
                                   dtype=nl.float32,
                                   buffer=nl.sbuf)
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          nisa.tensor_copy(
            dst=result_packed[0:TILE_M, bn * TILE_N:(bn + 1) * TILE_N],
            src=result_tmps[m][bm][bn][0:TILE_M, 0:TILE_N])

        # Copy packed result from SBUF to HBM.
        nisa.dma_copy(dst=result[(TILES_IN_BLOCK_M * m + bm) *
                                 TILE_M:(TILES_IN_BLOCK_M * m + bm + 1) * TILE_M,
                                 BLOCK_N * n:BLOCK_N * (n + 1)],
                      src=result_packed[0:TILE_M, 0:BLOCK_N])

  return result
```

## nki-language-guide.html

SUMMARY: This document covers the NKI language for writing optimized kernel functions on AWS Trainium devices, demonstrating the compilation model, tensor operations, control flow, and composable kernel patterns through practical API usage examples.

```python
import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit
def nki_tensor_add_kernel(a_input, b_input):
    """
    NKI kernel to compute element-wise addition of two input tensors.
    """
    assert a_input.shape == b_input.shape
    assert a_input.dtype == b_input.dtype

    print(f"adding tensors of type {a_input.dtype} and shape {a_input.shape}")

    assert a_input.shape[0] <= nl.tile_size.pmax

    a_tile = nl.ndarray(shape=a_input.shape, dtype=a_input.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_tile, src=a_input)

    b_tile = nl.ndarray(shape=b_input.shape, dtype=b_input.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_tile, src=b_input)

    c_tile = nl.ndarray(shape=a_input.shape, dtype=a_input.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

    c_output = nl.ndarray(dtype=a_input.dtype, shape=a_input.shape, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=c_output, src=c_tile)

    return c_output
```

```python
@nki.jit
def example_kernel(a_input):
    for i in range(4):
        tile = nl.ndarray((128, 512), dtype=nl.float16, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile, src=a_input[i * 128:(i + 1) * 128, :])
        if i % 2 == 0:
            nisa.tensor_scalar(dst=tile, data=tile, op0=nl.add, operand0=1.0)
```

```python
l = [1,2,3]
l.append(4.1)
l.extend(("Hello", "List"))
size = len(l)
third = l[2]

if l.index(2):
    print("list contains 2")

l.remove(1)
l.reverse()
for x in l:
    print(x)
```

```python
d = dict()
d['a'] = 1

print(d.keys())
print(d.items())

for k in d.keys():
    v = d[k]
    print(k, v)

if d.pop('a'):
    print("removed 'a' from dictionary")

a = d.setdefault('a', 2)
```

```python
int_tensor = nl.ndarray((128, 256), dtype=nl.int32, buffer=nl.sbuf)
float_tensor = int_tensor.view(nl.float32)
```

```python
t = nl.ndarray((128, 1024), dtype=nl.float16, buffer=nl.sbuf)
u = t.ap(pattern=[(1, 128), (2, 512)])
```

```python
t = nl.ndarray((128,128), nl.float16, nl.sbuf)
assert t.shape == (128,128)
assert t.dtype == nl.float16
assert t.buffer == nl.sbuf
```

```python
t = nl.ndarray((128,128), nl.float16, nl.sbuf, name="my_weights")
```

```python
u = t.reshape((128,2,64))
v = t.reshape((128,32,4))
```

```python
u = t[0,0,10]
u = t[0,1,0]
u = t[63,63,63]

u = t[0:64, 0, 0:64]
u = t[:, 0, :]
u = t[:, :, ::2]

u = t[...]
u = t[:,...]
u = t[0,...,:]

u = t[0,...]
assert u.shape == (64,64)

v = u[0:32, :]
assert v.shape == (32, 64)

u = t[0,...]
print(u.offset)
print(u.get_pattern())

u = t.ap(offset=0, pattern=[...])
```

```python
inputs = [a, b, c]
outputs = [x, y, z]

assert len(inputs) == len(outputs)
for i in range(len(inputs)):
    if i % 2 == 0:
        nisa.nc_transpose(dst=outputs[i], data=inputs[i])
    else:
        nisa.reciprocal(dst=outputs[i], data=inputs[i])
```

```python
l = [1,2,3]
for x in l:
    print(x)

t = (1,2,3)
for x in t:
    print(x)
```

```python
x = 0
while x < 10:
    print(x)
    x += 1
```

```python
for i in nki.isa.dynamic_range(10):
    process_tensor(t[i])
```

```python
count = nki.isa.register_alloc(0)
nisa.register_load(count, count_tensor)
for i in nki.isa.dynamic_range(count):
    process_tensor(t[i])
```

```python
cond = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)

reg = nisa.register_alloc(1)

while reg:
    nisa.dma_copy(dst=cond, ...)
    nisa.register_load(reg, cond)
```

```python
from dataclasses import dataclass
from nki.language import NKIObject

@dataclass
class C(NKIObject):
    x: int
    y: bool = False

    def toggle(self):
        self.y = not self.y

c = C(1)
c.toggle()
print(c.x, c.y)
```

```python
from nki.language import NKIObject

class A(NKIObject):
    x: int = 1
    def __init__(self, x):
        self.x = x

@nki.jit
def kernel(a: A): ...

kernel(A(1))
```

```python
from enum import Enum

class E(Enum):
    x = 1
    y = 2
    z = 3

def f(e: E):
    if e == E.x: ...
    elif e == E.y: ...
    elif e == E.z: ...

f(E.x)
```

```python
def tiled_process(input_tensor, output_tensor, tile_fn):
    """Generic kernel that applies tile_fn to each tile of the input."""
    for i in range(input_tensor.shape[0] // nl.tile_size.pmax):
        tile = nl.ndarray((128, 512), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile, src=input_tensor[i * 128:(i + 1) * 128, :])

        result = nl.ndarray((128, 512), dtype=input_tensor.dtype, buffer=nl.sbuf)
        tile_fn(dst=result, src=tile)

        nisa.dma_copy(dst=output_tensor[i * 128:(i + 1) * 128, :], src=result)

def my_activation(dst, src):
    nisa.activation(dst=dst, data=src, op=nl.relu)

def my_scale(dst, src):
    nisa.tensor_scalar(dst=dst, data=src, op0=nl.multiply, operand0=0.5)

@nki.jit
def relu_kernel(a_input, a_output):
    tiled_process(a_input, a_output, my_activation)

@nki.jit
def scale_kernel(a_input, a_output):
    tiled_process(a_input, a_output, my_scale)
```

```python
from nki.compiler.kernel_builder import compile_kernel

compile_kernel(
    tiled_process,
    inputs={"input_tensor": input_array},
    outputs={"output_tensor": output_array},
    compile_opts=opts,
    tile_fn=my_activation,
)
```

```python
def select_activation(name):
    if name == "relu":
        return my_relu
    elif name == "gelu":
        return my_gelu

@nki.jit
def kernel(a_input, a_output):
    act_fn = select_activation("relu")
    act_fn(dst=a_output, src=a_input)
```

## quickstart-implement-run-kernel.html

SUMMARY: This document demonstrates how to write a basic NKI kernel for AWS Trainium accelerators using the Neuron Kernel Interface, showing the complete workflow of reading inputs from HBM, performing computation in on-chip SBUF memory, and writing results back to HBM.

```python
import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit
def nki_tensor_add_kernel(a_input, b_input):
    """
    NKI kernel to compute element-wise addition of two input tensors.
    """

    # check both input tensor shapes/dtypes are the same for element-wise operation.
    assert a_input.shape == b_input.shape
    assert a_input.dtype == b_input.dtype

    # Check the first dimension's size to ensure it does not exceed on-chip
    # memory tile size, since this simple kernel does not tile inputs.
    assert a_input.shape[0] <= nl.tile_size.pmax

    # Allocate space for the input tensors in SBUF and copy the inputs from HBM
    # to SBUF with DMA copy.
    a_tile = nl.ndarray(shape=a_input.shape, dtype=a_input.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_tile, src=a_input)

    b_tile = nl.ndarray(shape=b_input.shape, dtype=b_input.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_tile, src=b_input)

    # Allocate space for the result and use tensor_tensor to perform
    # element-wise addition. Note: the first argument of 'tensor_tensor'
    # is the destination tensor.
    c_tile = nl.ndarray(shape=a_input.shape, dtype=a_input.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

    # Create a tensor in HBM and copy the result into HBM.
    c_output = nl.ndarray(dtype=a_input.dtype, shape=a_input.shape, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=c_output, src=c_tile)

    # Return kernel output as function output.
    return c_output
```

## indexing-overview.html

SUMMARY: This document covers tensor indexing techniques in NKI kernels, demonstrating how to use integer indexing, slicing, and advanced memory access patterns (including striding and reshaping) to efficiently manipulate tensors on NeuronCore devices.

```python
import nki
import nki.language as nl
import math

@nki.jit
def tensor_split_kernel_(in_tensor):
  """NKI kernel to split an input tensor into two output tensors, along the column axis.

  The even columns of the input tensor will be gathered into the first output tensor,
  and the odd columns of the input tensor will be gathered into the second output tensor.

  Args:
      in_tensor: an input tensor
  Returns:
      out_tensor_even: a first output tensor (will hold the even columns of the input tensor)
      out_tensor_odd: a second output tensor (will hold the odd columns of the input tensor)
  """

  # This example only works for tensors with a partition dimension that fits in the SBUF
  assert in_tensor.shape[0] <= nl.tile_size.pmax

  # Extract tile sizes.
  sz_p, sz_f = in_tensor.shape
  sz_fout_even = sz_f - sz_f // 2
  sz_fout_odd = sz_f // 2

  # create output tensors
  out_tensor_even = nl.ndarray((sz_p, sz_fout_even), dtype=in_tensor.dtype, buffer=nl.shared_hbm)
  out_tensor_odd = nl.ndarray((sz_p, sz_fout_odd), dtype=in_tensor.dtype, buffer=nl.shared_hbm)

  # Load input data from external memory to on-chip memory
  in_tile = nl.load(in_tensor)

  # Store the results back to external memory
  nl.store(out_tensor_even, value=in_tile[:, 0:sz_f:2])
  nl.store(out_tensor_odd,  value=in_tile[:, 1:sz_f:2])

  return out_tensor_even, out_tensor_odd
```

```python
import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit
def tensor_transpose2D_kernel_(in_tensor, shape2D):
  """
  NKI kernel to reorder the elements on axis[1] of the input tensor.

  Every row of the input tensor is a flattened row-major 2D matrix.
  The shape2D argument defines the dimensions of the flattened matrices (#rows,#cols).
  Our goal in this kernel is to transpose these flattened 2D matrices, i.e. make them (#cols,#rows).

  Args:
    in_tensor: an input tensor
    shape2D: tuple representing the dimensions to be transposed: (#rows, #cols)
  """
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # Gather input shapes
  sz_p, _ = in_tensor.shape

  # Load input data from external memory to on-chip memory
  in_tile = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=in_tile, src=in_tensor)

  # Performing f1/f2 transpose
  # ==========================
  # The desired transpose pattern is provided as an input:
  sz_f1, sz_f2 = shape2D

  # Perform the transposition via element-wise SBUF-to-SBUF copies
  # with index arithmetic to scatter elements into transposed positions.
  # RHS traverses an F1 x F2 matrix in row major order
  # LHS traverses an F2 x F1 (transposed) matrix in row major order
  out_tile = nl.ndarray(shape=(sz_p, sz_f2*sz_f1), dtype=in_tensor.dtype,
                        buffer=nl.sbuf)
  for i_f1 in nl.affine_range(sz_f1):
    for i_f2 in nl.affine_range(sz_f2):
      nisa.tensor_copy(dst=out_tile[:, nl.ds(i_f2*sz_f1+i_f1, 1)],
                       src=in_tile[:, nl.ds(i_f1*sz_f2+i_f2, 1)])

  # Finally, we store out_tile to external memory
  nisa.dma_copy(dst=out_tensor, src=out_tile)

  return out_tensor
```

```python
import nki
import nki.language as nl

@nki.jit
def tensor_maxpool_kernel_(in_tensor, sz_pool):
  """NKI kernel to compute a 2D max-pool operation

  Args:
      in_tensor: an input tensor, of dimensions C x H x W
      sz_pool: integer P representing a (square) pool-window size
  Returns:
      out_tensor: the resulting output tensor, of dimensions C x (H/P) x (W/P)
  """

  # Get input/output dimensions
  sz_p, sz_hin, sz_win = in_tensor.shape
  sz_hout, sz_wout = sz_hin // sz_pool, sz_win // sz_pool
  out_tensor = nl.ndarray((sz_p, sz_hout, sz_wout), dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  # Load input data from external memory to on-chip memory
  in_tile = nl.load(in_tensor)

  # Perform the pooling operation using an access pattern to create a 5D view:
  # [sz_p, sz_hout, sz_wout, sz_pool, sz_pool]
  # The pool dimensions are placed last so we can reduce over them.
  pool_view = in_tile.ap([
    [sz_hin * sz_win, sz_p],      # partition stride
    [sz_pool * sz_win, sz_hout],   # outer row stride (hop by pool rows)
    [sz_pool, sz_wout],            # outer col stride (hop by pool cols)
    [sz_win, sz_pool],             # inner row stride (within pool window)
    [1, sz_pool],                  # inner col stride (within pool window)
  ])
  out_tile = nl.max(pool_view, axis=[3, 4])

  # Store the results back to external memory
  nl.store(out_tensor, value=out_tile)

  return out_tensor
```

## nki_block_dimension_migration_guide.html

SUMMARY: This document explains how to migrate NKI code from using block dimensions (a removed software concept) to alternative approaches, demonstrating tensor allocation patterns, loop restructuring, and multi-buffering techniques for SBUF tensors on Trainium accelerators.

```python
# Example 1: Basic tensor with block dimensions (old pattern)
a = nl.ndarray((4, 8, nl.par_dim(128), 2, 512), buffer=nl.sbuf)
# - (4, 8): (B) block dimensions
# - 128: (P) partition dimension
# - (2, 512): (F) free dimension
```

```python
# Example 2: Migration - blocks need to be alive simultaneously (move to free dimension)
@nki.jit
def sb_blocks(inp):
    res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
    add_buf = nl.ndarray(shape=(8, nl.par_dim(128), 512), dtype=inp.dtype, buffer=nl.sbuf)
    for i in range(8):
        add_buf[i] = nl.load(inp[i])
    for i in range(8):
        nl.store(res[i], add_buf[i])
    return res

# should migrate to
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
# Example 3: Migration - blocks don't need to be alive simultaneously (hoist down)
@nki.jit
def sb_blocks(inp):
    res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
    add_buf = nl.ndarray(shape=(8, nl.par_dim(128), 512), dtype=inp.dtype, buffer=nl.sbuf)
    for i in range(8):
        add_buf[i] = nl.load(inp[i])
        nl.store(res[i], add_buf[i])
    return res

# should migrate to
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
# Example 4: Multi-buffering with direct allocation (old pattern with block dimensions)
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
# Example 5: Multi-buffering after block dimension removal (new pattern)
def interleave_alloc_func(idx, pdim_size, fdim_size):
    assert idx == ()  # No block dimension
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

## average_pool2d.html

SUMMARY: This document demonstrates how to implement a 2D average pooling operation using NKI, showcasing multi-dimensional memory access patterns, access pattern views for strided tensor operations, and dimensionality reduction techniques on AWS Trainium accelerators.

```python
import nki
import nki.isa as nisa
import nki.language as nl
from nki.typing import tensor

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

  # Generate pool access pattern to create a 5D view:
  # [sz_p, sz_hout, sz_wout, sz_pool, sz_pool]
  # The pool dimensions are placed last so we can reduce over them.

  # Load input data from external memory to on-chip memory
  in_tile = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=in_tile, src=in_tensor)

  # Perform the pooling operation using an access pattern view:
  # The .ap() creates a strided 5D view of the 3D input tile,
  # grouping elements into pool windows for reduction.
  pool_view = in_tile.ap([
    [sz_hin * sz_win, sz_p],      # partition stride
    [sz_pool * sz_win, sz_hin // sz_pool],  # outer row stride
    [sz_pool, sz_win // sz_pool],            # outer col stride
    [sz_win, sz_pool],             # inner row stride (within pool window)
    [1, sz_pool],                  # inner col stride (within pool window)
  ])
  sum_tile = nl.sum(pool_view, axis=[3, 4])
  out_tile = nl.ndarray(sum_tile.shape, dtype=sum_tile.dtype, buffer=nl.sbuf)
  nisa.tensor_scalar(dst=out_tile, data=sum_tile, op0=nl.multiply,
                     operand0=1.0 / (pool_size * pool_size))

  # Store the results back to hbm
  nisa.dma_copy(dst=out_tensor, src=out_tile)

  # Transfer the ownership of `out_tensor` to the caller
  return out_tensor
```

## transpose2d.html

SUMMARY: This document demonstrates how to transpose a 2D tensor along two free-dimension axes in NKI using element-wise copies with index arithmetic, covering NKI syntax, programming model, and multi-dimensional memory addressing patterns.

```python
import nki
import nki.language as nl
import nki.isa as nisa


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
  """
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # Gather input shapes
  sz_p, _ = in_tensor.shape

  # Load input data from external memory to on-chip memory
  in_tile = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=in_tile, src=in_tensor)

  # Performing f1/f2 transpose
  # ==========================
  # The desired transpose pattern is provided as an input:
  sz_f1, sz_f2 = shape2D

  # Perform the transposition via element-wise SBUF-to-SBUF copies
  # with index arithmetic to scatter elements into transposed positions.
  # RHS traverses an F1 x F2 matrix in row major order
  # LHS traverses an F2 x F1 (transposed) matrix in row major order
  out_tile = nl.ndarray(shape=(sz_p, sz_f2*sz_f1), dtype=in_tensor.dtype,
                        buffer=nl.sbuf)
  for i_f1 in nl.affine_range(sz_f1):
    for i_f2 in nl.affine_range(sz_f2):
      nisa.tensor_copy(dst=out_tile[:, nl.ds(i_f2*sz_f1+i_f1, 1)],
                       src=in_tile[:, nl.ds(i_f1*sz_f2+i_f2, 1)])

  # Finally, we store out_tile to external memory
  nisa.dma_copy(dst=out_tensor, src=out_tile)

  return out_tensor
```

## nki-dynamic-loops.html

SUMMARY: This document covers the `dynamic_range` NKI API for creating on-chip loops with runtime-determined bounds, contrasting it with compile-time range iterators and demonstrating how to use VirtualRegisters for dynamic trip counts.

```python
import nki.language as nl
import nki.isa as nisa

for _ in nl.dynamic_range(1):
    tile = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
    result = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(src=input_tensor[0:128, 0:512], dst=tile)
    nisa.tensor_tensor(dst=result, data1=tile, data2=tile, op=nl.multiply)
    nisa.dma_copy(src=result, dst=out_tensor[0:128, 0:512])
```

```python
import nki.language as nl
import nki.isa as nisa

start = nisa.register_alloc(0)
stop = nisa.register_alloc(512)
for i in nl.dynamic_range(start, stop, 128):
    tile = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
    result = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(src=input_tensor.ap([[512, 128], [1, 512]], scalar_offset=i), dst=tile)
    nisa.tensor_scalar(dst=result, data=tile, op0=nl.add, operand0=2.0)
    nisa.dma_copy(src=result, dst=out_tensor.ap([[512, 128], [1, 512]], scalar_offset=i))
```

```python
import nki.language as nl
import nki.isa as nisa

begin = nisa.register_alloc(0)
end = nisa.register_alloc(4)
for i in nl.dynamic_range(begin, end, 2):
    pass
```

```python
import nki.language as nl
import nki.isa as nisa

reg = nisa.register_alloc(1)
while reg:
    # perform work ...
    nisa.register_load(reg, cond_tensor)
```

## lnc.html

SUMMARY: This document covers how to use Logical Neuron Cores (LNC) in NKI kernels to run computations on multiple physical cores simultaneously, demonstrating the num_programs() and program_id() APIs for core-aware kernel programming.

```python
@nki.jit
def lnc_test(input):
    if nl.num_programs() > 1:
        print("Running on multiple cores")
    else:
        print("Running on one core - no LNC")
```

```python
def lnc_test(input):
    # Check the first dimension is 2 for this example
    assert input.shape[0] == 2

    # create temporary storage on SBUF for comptation
    in_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)
    out_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)

    # create output tensor
    output = nl.ndarray(input.shape, input.dtype, buffer=nl.shared_hbm)

    if nl.num_programs() == 1:
        # Not using multiple cores, process two tiles
        for i in range(2):
            nisa.dma_copy(in_tile, input[i])
            nisa.reciprocal(out_tile, in_tile)
            nisa.dma_copy(output[i], out_tile)
    else:
        # Using multiple cores, process tiles in parallel, one per core
        i = nl.program_id(0)
        nisa.dma_copy(in_tile, input[i])
        nisa.reciprocal(out_tile, in_tile)
        nisa.dma_copy(output[i], out_tile)
    return output
```

## nki.jit.html

SUMMARY: This document demonstrates the `@nki.jit` decorator for compiling NKI functions to run on NeuronDevices, showing how to write a custom tensor addition kernel using NKI primitives for on-chip memory management and tensor operations.

```python
@nki.jit()
def nki_tensor_add_kernel(a_input, b_input):
    # Check both input tensor shapes are the same for element-wise operation.
    assert a_input.shape == b_input.shape

    # Check the first dimension's size to ensure it does not exceed on-chip
    # memory tile size, since this simple kernel does not tile inputs.
    assert a_input.shape[0] <= nl.tile_size.pmax

    # Allocate space for the input tensors in SBUF and copy the inputs from HBM
    # to SBUF with DMA copy.
    a_tile = nl.ndarray(dtype=a_input.dtype, shape=a_input.shape, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_tile, src=a_input)

    b_tile = nl.ndarray(dtype=b_input.dtype, shape=b_input.shape, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_tile, src=b_input)

    # Allocate space for the result and use tensor_tensor to perform
    # element-wise addition. Note: the first argument of 'tensor_tensor'
    # is the destination tensor.
    c_tile = nl.ndarray(dtype=a_input.dtype, shape=a_input.shape, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

    # Create a tensor in HBM and copy the result into HBM.
    c_output = nl.ndarray(dtype=a_input.dtype, shape=a_input.shape, buffer=nl.hbm)
    nisa.dma_copy(dst=c_output, src=c_tile)

    # Return kernel output as function output.
    return c_output
```
# nkilib production kernel examples

Real production NKI 0.3 kernels from `nkilib`. Useful as reference patterns for: HBM<->SBUF DMA staging, multi-buffered tile loops, mxfp4 dequant + nc_matmul_mx, gather/scatter for routed dispatch, RoPE-style fused elementwise compute, and SbufManager-based allocation.

## nkilib/core/embeddings/rope.py

```python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Rotary Position Embedding (RoPE) kernels for NeuronCore."""

import nki
import nki.isa as nisa
import nki.language as nl

from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import get_verified_program_sharding_info
from ..utils.tensor_view import TensorView


@nki.jit
def RoPE(
    x_in: nl.ndarray,
    cos: nl.ndarray,
    sin: nl.ndarray,
    lnc_shard: bool = False,
    contiguous_layout: bool = True,
    relayout_in_sbuf: bool = False,
) -> nl.ndarray:
    """
    Apply Rotary Position Embedding (RoPE) to input embeddings.

    Standalone kernel with HBM I/O and optional LNC sharding.
    Supports both contiguous and interleaved memory layouts with automatic
    layout conversion via strided DMA or SBUF matmul.

    Dimensions:
        d_head: Head dimension (64 or 128)
        B: Batch size
        n_heads: Number of attention heads
        S: Sequence length (divisible by n_prgs if lnc_shard=True)

    Args:
        x_in (nl.ndarray): [d_head, B, n_heads, S] @ HBM, Input embeddings
        cos (nl.ndarray): [d_head//2, B, S] @ HBM, Cosine frequencies
        sin (nl.ndarray): [d_head//2, B, S] @ HBM, Sine frequencies
        lnc_shard (bool): Parallelize across LNC cores by tiling sequence dimension
        contiguous_layout (bool): Memory layout in d_head dimension.
            True: [first_half, second_half] (default, more efficient).
            False: [even, odd, even, odd, ...] (interleaved)
        relayout_in_sbuf (bool): Use SBUF matmul for layout conversion (only for small tensors)

    Returns:
        output (nl.ndarray): [d_head, B, n_heads, S] @ HBM, RoPE applied output

    Notes:
        - SBUF size constraint (for bf16): B * n_heads * S <= 73728 (approximately 72K).
          This limit applies regardless of d_head. Exceeding this limit will
          cause compilation failure. For larger sizes, tile the computation.
        - When relayout_in_sbuf=True with interleaved layout, a stricter limit
          applies: B * n_heads * S <= gemm_moving_fmax (typically 512)
        - d_head must be even (pairs of elements are rotated)
        - When lnc_shard=True, S must be divisible by number of programs
        - For large tensors with interleaved layout, uses strided DMA

    Pseudocode:
        # Determine sharding and tile size
        tile_size = S // n_prgs
        tile_start = tile_size * prg_id

        # Load input tile to SBUF (with optional layout conversion)
        if is_dma_relayout:
            x_in_sb = load_strided(x_in, even_odd_separated)
        else:
            x_in_sb = load_contiguous(x_in)

        # Load cos/sin frequency tiles
        cos_sb = load(cos[tile_start:tile_start+tile_size])
        sin_sb = load(sin[tile_start:tile_start+tile_size])

        # Apply RoPE rotation in SBUF
        x_out_sb = rope_sbuf(x_in_sb, cos_sb, sin_sb)

        # Store output (with optional layout conversion)
        if is_dma_relayout:
            store_strided(x_out, x_out_sb, even_odd_interleaved)
        else:
            store_contiguous(x_out, x_out_sb)
    """

    _validate_rope_inputs(x_in, cos, sin, 'RoPE')

    d_head, B, n_heads, S = x_in.shape
    half_d = d_head // 2

    # Determine parallelization across LNC cores
    n_prgs, prg_id = 1, 0
    if lnc_shard:
        _, n_prgs, prg_id = get_verified_program_sharding_info("RoPE", (0, 1))

    # Tile along sequence dimension
    kernel_assert(S % n_prgs == 0, f'RoPE: sequence length {S} not divisible by {n_prgs} programs')
    tile_size = S // n_prgs
    tile_start = tile_size * prg_id

    # Determine layout conversion strategy: DMA (strided access) vs SBUF (matmul)
    # SBUF relayout limited by gemm_moving_fmax, fallback to DMA for large tensors
    is_relayout_in_sbuf_supported = B * n_heads * S <= nl.tile_size.gemm_moving_fmax
    is_dma_relayout = not contiguous_layout and (not relayout_in_sbuf or not is_relayout_in_sbuf_supported)
    is_relayout_in_sbuf = not contiguous_layout and relayout_in_sbuf and is_relayout_in_sbuf_supported

    # Load input to SBUF with optional layout conversion via strided DMA
    x_in_sb = nl.ndarray((d_head, B, n_heads, tile_size), dtype=x_in.dtype, buffer=nl.sbuf)
    if is_dma_relayout:
        # Gather even/odd indices with stride=2 in d_head dimension
        nisa.dma_copy(
            dst=x_in_sb[:half_d, :, :, :],
            src=TensorView(x_in)
            .slice(dim=0, start=0, end=d_head, step=2)
            .slice(dim=3, start=tile_start, end=tile_start + tile_size)
            .get_view(),
        )
        nisa.dma_copy(
            dst=x_in_sb[half_d:, :, :, :],
            src=TensorView(x_in)
            .slice(dim=0, start=1, end=d_head, step=2)
            .slice(dim=3, start=tile_start, end=tile_start + tile_size)
            .get_view(),
        )
    else:
        nisa.dma_copy(dst=x_in_sb, src=x_in[:, :, :, tile_start : tile_start + tile_size])

    # Load cos/sin frequency tiles
    cos_sb = nl.ndarray((half_d, B, tile_size), dtype=cos.dtype, buffer=nl.sbuf)
    sin_sb = nl.ndarray((half_d, B, tile_size), dtype=sin.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=cos_sb, src=cos[:, :, tile_start : tile_start + tile_size])
    nisa.dma_copy(dst=sin_sb, src=sin[:, :, tile_start : tile_start + tile_size])

    # Compute RoPE rotation in SBUF
    x_out_sb = nl.ndarray(x_in_sb.shape, dtype=x_in_sb.dtype, buffer=nl.sbuf)
    RoPE_sbuf(x_in_sb, cos_sb, sin_sb, x_out_sb, convert_from_interleaved=is_relayout_in_sbuf)

    # Store output to HBM with optional layout conversion via strided DMA
    x_out = nl.ndarray(x_in.shape, dtype=x_in.dtype, buffer=nl.shared_hbm)
    if is_dma_relayout:
        # Scatter even/odd indices with stride=2 in d_head dimension
        nisa.dma_copy(
            dst=TensorView(x_out)
            .slice(dim=0, start=0, end=d_head, step=2)
            .slice(dim=3, start=tile_start, end=tile_start + tile_size)
            .get_view(),
            src=x_out_sb[:half_d, :, :, :],
        )
        nisa.dma_copy(
            dst=TensorView(x_out)
            .slice(dim=0, start=1, end=d_head, step=2)
            .slice(dim=3, start=tile_start, end=tile_start + tile_size)
            .get_view(),
            src=x_out_sb[half_d:, :, :, :],
        )
    else:
        nisa.dma_copy(dst=x_out[:, :, :, tile_start : tile_start + tile_size], src=x_out_sb)

    return x_out


def RoPE_sbuf(
    x_in_sb: nl.ndarray,
    cos_sb: nl.ndarray,
    sin_sb: nl.ndarray,
    x_out_sb: nl.ndarray,
    convert_from_interleaved: bool = False,
) -> nl.ndarray:
    """
    Apply RoPE on tensors in SBUF (for megakernel fusion).
    Helper function that operates entirely in SBUF without HBM I/O.

    RoPE Formula:
        out[even] = x[even]*cos - x[odd]*sin
        out[odd] = x[odd]*cos + x[even]*sin

    Args:
        x_in_sb (nl.ndarray): [d_head, B, n_heads, S] @ SBUF - input embeddings
        cos_sb (nl.ndarray): [d_head//2, B, S] @ SBUF - cosine frequencies
        sin_sb (nl.ndarray): [d_head//2, B, S] @ SBUF - sine frequencies
        x_out_sb (nl.ndarray): [d_head, B, n_heads, S] @ SBUF - output buffer
        convert_from_interleaved (bool): convert from interleaved to contiguous layout

    Returns:
        nl.ndarray: x_out_sb with RoPE applied (modified in-place)

    Notes:
        - Assumes contiguous layout unless convert_from_interleaved=True
        - For large tensors with interleaved layout, use RoPE() with strided DMA
    """

    d_head, B, n_heads, S = x_out_sb.shape
    half_d = d_head // 2

    _validate_rope_inputs(x_in_sb, cos_sb, sin_sb, 'RoPE_sbuf')
    kernel_assert(x_in_sb.dtype == x_out_sb.dtype, 'RoPE_sbuf: dtype mismatch between x_in_sb and x_out_sb')

    # Convert interleaved to contiguous layout if needed
    if convert_from_interleaved:
        convert_to_interleaved_mat = _compute_convert_to_interleaved_mat(x_in_sb)
        x_in_sb = _convert_from_interleaved(x_in_sb, convert_to_interleaved_mat)

    # Copy odd half to separate buffer (required for tensor_tensor base partition alignment)
    sb_odd = nl.ndarray((half_d, B, n_heads, S), dtype=x_in_sb.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sb_odd, src=x_in_sb[half_d:, :, :, :])

    # Allocate buffers for intermediate products
    even_cos = nl.ndarray((half_d, B, n_heads, S), dtype=x_in_sb.dtype, buffer=nl.sbuf)
    odd_cos = nl.ndarray((half_d, B, n_heads, S), dtype=x_in_sb.dtype, buffer=nl.sbuf)
    even_sin = nl.ndarray((half_d, B, n_heads, S), dtype=x_in_sb.dtype, buffer=nl.sbuf)
    odd_sin = nl.ndarray((half_d, B, n_heads, S), dtype=x_in_sb.dtype, buffer=nl.sbuf)

    # Compute RoPE: out_even = even*cos - odd*sin, out_odd = odd*cos + even*sin
    # Use access patterns to broadcast cos/sin across n_heads dimension
    nisa.tensor_tensor(
        even_cos,
        x_in_sb[:half_d, :, :, :],
        TensorView(cos_sb).expand_dim(2).broadcast(dim=2, size=n_heads).get_view(),
        nl.multiply,
    )
    nisa.tensor_tensor(
        odd_cos,
        sb_odd[:half_d, :, :, :],
        TensorView(cos_sb).expand_dim(2).broadcast(dim=2, size=n_heads).get_view(),
        nl.multiply,
    )
    nisa.tensor_tensor(
        even_sin,
        x_in_sb[:half_d, :, :, :],
        TensorView(sin_sb).expand_dim(2).broadcast(dim=2, size=n_heads).get_view(),
        nl.multiply,
    )
    nisa.tensor_tensor(
        odd_sin,
        sb_odd[:half_d, :, :, :],
        TensorView(sin_sb).expand_dim(2).broadcast(dim=2, size=n_heads).get_view(),
        nl.multiply,
    )

    nisa.tensor_tensor(x_out_sb[:half_d, :, :, :], even_cos, odd_sin, nl.subtract)
    nisa.tensor_tensor(x_out_sb[half_d:, :, :, :], odd_cos, even_sin, nl.add)

    # Convert back to interleaved layout if needed
    if convert_from_interleaved:
        x_out_sb = _convert_to_interleaved(x_out_sb, convert_to_interleaved_mat)

    return x_out_sb


def _compute_convert_to_interleaved_mat(x_sb: nl.ndarray) -> nl.ndarray:
    """
    Generate permutation matrix for RoPE layout conversion.

    Creates matrix P for converting between contiguous and interleaved layouts.
    P @ X transforms [e0,e1,...,o0,o1,...] to [e0,o0,e1,o1,...].
    P^T @ X transforms [e0,o0,e1,o1,...] to [e0,e1,...,o0,o1,...].

    Args:
        x_sb (nl.ndarray): [d_head, B, n_heads, S] @ SBUF, Input tensor for shape info

    Returns:
        nl.ndarray: [d_head, d_head] @ SBUF, Permutation matrix

    Notes:
        - Only supports tensors where B*n_heads*S ≤ gemm_moving_fmax
        - d_head must be even
        - Uses strided access on identity matrix to build permutation
    """
    d_head, B, n_heads, S = x_sb.shape
    half_d = d_head // 2
    kernel_assert(d_head % 2 == 0, f'_compute_convert_to_interleaved_mat: d_head must be even, got {d_head}')

    identity_sb = nl.shared_identity_matrix(d_head, dtype=x_sb.dtype)

    """
    Extract permutation via strided access pattern.
    
    Pattern [[d_head, d_head], [1, 2], [2, half_d]] reads identity with stride=2 in innermost dim.
    For each row i: reads [i[0], i[2], i[4], ...] then [i[1], i[3], i[5], ...].
    Destination reshape (d_head, 2, half_d) writes: row i -> [[even_cols], [odd_cols]].
    Result: even rows get 1 in first half, odd rows get 1 in second half.
    This creates P where P@X transforms [e0,e1,...,o0,o1,...] -> [e0,o0,e1,o1,...].
    """
    convert_to_interleaved_mat = nl.ndarray((d_head, d_head), dtype=x_sb.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=convert_to_interleaved_mat.reshape((d_head, 2, half_d)),
        src=identity_sb.ap(pattern=[[d_head, d_head], [1, 2], [2, half_d]]),
        engine=nisa.scalar_engine,
    )

    return convert_to_interleaved_mat


def _convert_from_interleaved(x_sb: nl.ndarray, convert_to_interleaved_mat: nl.ndarray) -> nl.ndarray:
    """
    Convert interleaved to contiguous layout using matrix multiplication.

    Transforms [e0,o0,e1,o1,...] to [e0,e1,...,o0,o1,...] via P^T @ x_sb.

    Args:
        x_sb (nl.ndarray): [d_head, B, n_heads, S] @ SBUF, Input in interleaved layout
        convert_to_interleaved_mat (nl.ndarray): [d_head, d_head] @ SBUF, Permutation matrix

    Returns:
        nl.ndarray: [d_head, B, n_heads, S] @ SBUF, Output in contiguous layout

    Notes:
        - Returns new buffer (does not modify input)
    """
    d_head, B, n_heads, S = x_sb.shape
    kernel_assert(x_sb.buffer == nl.sbuf, '_convert_from_interleaved: input must be in SBUF')

    total_free_dim = B * n_heads * S
    fmax = nl.tile_size.gemm_moving_fmax
    x_converted_sb = nl.ndarray(x_sb.shape, dtype=x_sb.dtype, buffer=nl.sbuf)
    x_flat = x_sb.reshape((d_head, total_free_dim))
    x_out_flat = x_converted_sb.reshape((d_head, total_free_dim))
    for t_start in range(0, total_free_dim, fmax):
        t_size = min(fmax, total_free_dim - t_start)
        x_psum = nl.ndarray((d_head, t_size), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=x_psum, stationary=convert_to_interleaved_mat, moving=x_flat[:, nl.ds(t_start, t_size)])
        nisa.activation(dst=x_out_flat[:, nl.ds(t_start, t_size)], op=nl.copy, data=x_psum)
    return x_converted_sb


def _convert_to_interleaved(x_sb: nl.ndarray, convert_to_interleaved_mat: nl.ndarray) -> nl.ndarray:
    """
    Convert contiguous to interleaved layout using matrix multiplication.

    Transforms [e0,e1,...,o0,o1,...] to [e0,o0,e1,o1,...] via P @ x_sb.

    Args:
        x_sb (nl.ndarray): [d_head, B, n_heads, S] @ SBUF, Input in contiguous layout
        convert_to_interleaved_mat (nl.ndarray): [d_head, d_head] @ SBUF, Permutation matrix

    Returns:
        nl.ndarray: [d_head, B, n_heads, S] @ SBUF, Output in interleaved layout

    Notes:
        - Pre-transposes matrix to compensate for nc_matmul's implicit transpose
        - Modifies input buffer in-place
    """
    d_head, B, n_heads, S = x_sb.shape
    kernel_assert(x_sb.buffer == nl.sbuf, '_convert_to_interleaved: input must be in SBUF')

    # Pre-transpose to compensate for nc_matmul's implicit transpose
    convert_from_interleaved_sb = nl.ndarray((d_head, d_head), dtype=convert_to_interleaved_mat.dtype, buffer=nl.sbuf)
    convert_from_interleaved_psum = nl.ndarray((d_head, d_head), dtype=convert_to_interleaved_mat.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=convert_from_interleaved_psum, data=convert_to_interleaved_mat)
    nisa.tensor_copy(dst=convert_from_interleaved_sb, src=convert_from_interleaved_psum, engine=nisa.scalar_engine)

    total_free_dim = B * n_heads * S
    fmax = nl.tile_size.gemm_moving_fmax
    x_flat = x_sb.reshape((d_head, total_free_dim))
    for t_start in range(0, total_free_dim, fmax):
        t_size = min(fmax, total_free_dim - t_start)
        x_psum = nl.ndarray((d_head, t_size), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=x_psum, stationary=convert_from_interleaved_sb, moving=x_flat[:, nl.ds(t_start, t_size)])
        nisa.tensor_copy(dst=x_flat[:, nl.ds(t_start, t_size)], src=x_psum, engine=nisa.scalar_engine)
    return x_sb


def _validate_rope_inputs(x_in: nl.ndarray, cos: nl.ndarray, sin: nl.ndarray, func_name: str) -> None:
    """
    Validate RoPE input tensor shapes and constraints.

    Args:
        x_in (nl.ndarray): [d_head, B, n_heads, S], Input embeddings
        cos (nl.ndarray): [d_head//2, B, S], Cosine frequencies
        sin (nl.ndarray): [d_head//2, B, S], Sine frequencies
        func_name (str): Name of calling function for error messages

    Returns:
        None

    Notes:
        - Validates d_head in {64, 128}
        - Validates B in (0, 64]
        - Validates S in (0, 512]
        - Validates n_heads in (0, 16]
        - Validates cos/sin shapes match expected dimensions
    """
    d_head, B, n_heads, S = x_in.shape
    half_d = d_head // 2

    kernel_assert(d_head in (64, 128), f'{func_name}: d_head must be 64 or 128, got {d_head}')
    kernel_assert(B > 0, f'{func_name}: B must be > 0, got {B}')
    kernel_assert(S > 0, f'{func_name}: S must be > 0, got {S}')
    kernel_assert(n_heads > 0, f'{func_name}: n_heads must be > 0, got {n_heads}')

    kernel_assert(
        tuple(cos.shape) == (half_d, B, S),
        f'{func_name}: cos.shape expected ({half_d},{B},{S}), got {cos.shape}',
    )
    kernel_assert(
        tuple(sin.shape) == (half_d, B, S),
        f'{func_name}: sin.shape expected ({half_d},{B},{S}), got {sin.shape}',
    )
    kernel_assert(cos.dtype == sin.dtype, f'{func_name}: cos/sin dtype mismatch')
```

## nkilib/experimental/subkernels/topk_reduce.py

```python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MoE Top-K reduction across sparse all_to_all_v() collective output buffer."""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import get_verified_program_sharding_info
from ...core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast

_K_MAX = 8
_N_16BIT_ELEM_PER_INT32 = 2
_SUPPORTED_INPUT_DTYPES = [nl.bfloat16, nl.float16]


@nki.jit
def topk_reduce(
    input: nl.ndarray,
    T: int,
    K: int,
):
    """
    Compute MoE Top-K reduction across sparse all_to_all_v() collective output buffer.

    Gathers scattered rows by packed global token index and reduces along
    the K dimension. Supports LNC sharding on the H dimension.

    Dimensions:
        TK_padded: n_src_ranks * T, padded input row count
        H: Hidden dimension size (must be divisible by LNC)
        T: Total number of input tokens (up to 128)
        K: Number of routed experts per token (up to 8)

    Args:
        input (nl.ndarray): [TK_padded, H + 2]@HBM, bf16/fp16. Sparse input buffer containing T*K
            scattered outputs. Global token index is packed as int32 in the final 2x
            columns of each row.
        T (int): Total number of input tokens.
        K (int): Number of routed experts per token.

    Returns:
        output_hbm (nl.ndarray): [T, H]@HBM, bf16/fp16. Ordered and reduced output.

    Pseudocode:
        global_token_indices = extract_int32_index(input[:, H:])
        for token_idx in range(T):
            matching_rows = find_rows_where(global_token_indices == token_idx)
            output[token_idx] = sum(input[matching_rows, :H])
    """

    # Shapes, LNC sharding strategy
    _P_MAX = nl.tile_size.pmax
    TK_padded, H_padded = input.shape
    H = H_padded - _N_16BIT_ELEM_PER_INT32
    _, n_prgs, prg_id = get_verified_program_sharding_info("topk_reduce", (0, 1))
    H_local = H // n_prgs
    H_local_slice = nl.ds(H_local * prg_id, H_local)

    # Validation
    kernel_assert(
        input.dtype in _SUPPORTED_INPUT_DTYPES, f"Expected input.dtype in {_SUPPORTED_INPUT_DTYPES}, got {input.dtype=}"
    )
    kernel_assert(T <= _P_MAX, f"T must be <= {_P_MAX}")
    kernel_assert(K <= _K_MAX, f"K must be <= {_K_MAX}")
    kernel_assert(H % n_prgs == 0, f"Expected H divisible by LNC, got {H=} {n_prgs=}")

    # Allocations
    reduced_sb = nl.ndarray((T, H_local), dtype=input.dtype, buffer=nl.sbuf)
    global_token_indices_sb = nl.ndarray((T, TK_padded), dtype=nl.int32, buffer=nl.sbuf)
    output_hbm = nl.ndarray((T, H), dtype=input.dtype, buffer=nl.shared_hbm)

    # DMA transpose indices [TK_padded, 1] -> [1, TK_padded]
    nisa.dma_transpose(
        src=input.ap(
            pattern=[[H_padded // _N_16BIT_ELEM_PER_INT32, TK_padded], [1, 1], [1, 1], [1, 1]],
            offset=H // _N_16BIT_ELEM_PER_INT32,
            dtype=nl.int32,
        ),
        dst=global_token_indices_sb.ap(
            pattern=[[TK_padded, 1], [1, 1], [1, 1], [1, TK_padded]],
            offset=0,
        ),
    )

    # Broadcast [1, TK_padded] -> [T, TK_padded]
    # FIXME: (1) Move broadcast to DMA engines (2) LNC shard on tokens when T>32
    stream_shuffle_broadcast(global_token_indices_sb, global_token_indices_sb)

    # Find indices [T, K]
    arange_token_indices_T = nl.ndarray((T, _K_MAX), dtype=nl.uint32, buffer=nl.sbuf)
    gather_token_indices = nl.ndarray((T, _K_MAX), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.iota(
        pattern=[[0, _K_MAX]],
        offset=0,
        channel_multiplier=1,
        dst=arange_token_indices_T,
    )
    nisa.nc_find_index8(
        data=global_token_indices_sb,
        vals=arange_token_indices_T,
        dst=gather_token_indices,
    )

    # Use DMA + rmw add to reduce over topK
    for k_idx in range(K):
        src_access = input.ap(
            pattern=[[H, T], [1, H_local]],
            offset=H_local * prg_id,
            vector_offset=gather_token_indices.ap(
                pattern=[[_K_MAX, T], [1, 1]],
                offset=k_idx,
            ),
            indirect_dim=0,
        )

        if k_idx == 0:
            nisa.dma_copy(
                dst=reduced_sb[:, :],
                src=src_access,
            )
        else:
            nisa.dma_compute(
                dst=reduced_sb[:, :],
                srcs=[src_access, reduced_sb[:, :]],
                reduce_op=nl.add,
                unique_indices=True,
            )

    # Save reduced output — each core writes its H shard
    nisa.dma_copy(output_hbm[:, H_local_slice], reduced_sb)

    return output_hbm
```

## nkilib/core/moe/moe_tkg/gate_up_projection_mx.py

```python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gate/Up projection sub-kernels with LNC sharding support.

Supports multiple LNC sharding strategies (mutually exclusive):
- no sharding: Used when running without LNC (LNC=1).
- shard_on_I: Shard on I (intermediate) dimension. Default for most workloads.
- TODO: shard_on_T: Shard on T (token) dimension. Useful when T is large.

These sub-kernels can be used by any algorithm that requires LNC-sharded gate/up projection,
including all-expert, selective-load, or custom MoE implementations.
"""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

# Shared MX constants
from ...mlp.mlp_tkg.projection_mx_constants import (
    MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM,
    MIN_MATMULT_MX_P_DIM,
    SBUF_QUADRANT_SIZE,
    SCALE_P_ELEM_PER_QUADRANT,
    _psum_fmax,
    _q_height,
    _q_width,
    pad_to_valid_qmx_partitions,
)

# Common utils
from ...utils.common_types import ActFnType
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_nl_act_fn_from_type
from ...utils.tensor_view import TensorView


@nki.jit
def gate_up_projection_mx_shard_I(
    input_quant_sb: nl.ndarray,
    input_scale_sb: nl.ndarray,
    gate_weight_sb: nl.ndarray,
    up_weight_sb: nl.ndarray,
    gate_weight_scale_sb: nl.ndarray,
    up_weight_scale_sb: nl.ndarray,
    gate_bias_sb: Optional[nl.ndarray],
    up_bias_sb: Optional[nl.ndarray],
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    hidden_act_fn: ActFnType = ActFnType.Swish,
    activation_compute_dtype=nl.bfloat16,
) -> tuple[nl.ndarray, nl.ndarray]:
    """
    Compute gate and up projections with clamping, activation function, and MX quantization.

    When executed with LNC=2, inputs are expected to be sharded on I dimension and compute
    is sharded on I dimension.

    Usage:
        Tuned for: mx all-expert MoE algorithm
        Applicable to: any algorithm requiring mx I-sharded gate/up projection

    Args:
        input_quant_sb (nl.ndarray): [16_H * 8_H, H/512, T], Quantized input in SBUF (4_H packed in x4 dtype).
        input_scale_sb (nl.ndarray): [16_H * 8_H, H/512, T], Input scales in SBUF (in leading 4P of each quadrant).
        gate_weight_sb (nl.ndarray): [16_H * 8_H, H/512, I/512 * 4_I * 16_I * 8_I], Gate weights in SBUF
            (4_H packed in x4 dtype).
        up_weight_sb (nl.ndarray): [16_H * 8_H, H/512, I/512 * 4_I * 16_I * 8_I], Up weights in SBUF
            (4_H packed in x4 dtype).
        gate_weight_scale_sb (nl.ndarray): [16_H * 8_H, H/512, I/512 * 4_I * 16_I * 8_I], Gate weight scales
            in SBUF (in leading 4P of each quadrant).
        up_weight_scale_sb (nl.ndarray): [16_H * 8_H, H/512, I/512 * 4_I * 16_I * 8_I], Up weight scales
            in SBUF (in leading 4P of each quadrant).
        gate_bias_sb (Optional[nl.ndarray]): [16_I * 8_I, I/512, 4_I], Gate bias in SBUF.
        up_bias_sb (Optional[nl.ndarray]): [16_I * 8_I, I/512, 4_I], Up bias in SBUF.
        gate_clamp_upper_limit (Optional[float]): Upper clamp limit for gate projection.
        gate_clamp_lower_limit (Optional[float]): Lower clamp limit for gate projection.
        up_clamp_upper_limit (Optional[float]): Upper clamp limit for up projection.
        up_clamp_lower_limit (Optional[float]): Lower clamp limit for up projection.
        hidden_act_fn (ActFnType): Activation function type (default: Swish).
        activation_compute_dtype: Compute dtype for activations (default: bfloat16).

    Returns:
        out_quant_sb (nl.ndarray): [16_I * 8_I, I/512, T], Quantized output in SBUF (4_I packed in x4 dtype).
        out_scale_sb (nl.ndarray): [16_I * 8_I, I/512, T], Output scales in SBUF (in leading 4P of each quadrant).
    """

    # Step 1: Input validation
    TILE_H, n_H512_tiles, T = input_quant_sb.shape
    TILE_H_, n_H512_tiles_, I_local_padded = gate_weight_sb.shape
    I_local = I_local_padded
    kernel_assert(
        gate_weight_sb.shape == up_weight_sb.shape,
        f"expected gate and up weights to have the same shapes, got {gate_weight_sb.shape=}, {up_weight_sb.shape=}",
    )
    kernel_assert(
        gate_weight_scale_sb.shape == up_weight_scale_sb.shape,
        f"expected gate and up scales to have the same shapes, "
        f"got {gate_weight_scale_sb.shape=}, {up_weight_scale_sb.shape=}",
    )
    # Validate bias consistency: both must be None or both must have matching shapes
    if gate_bias_sb != None and up_bias_sb != None:
        kernel_assert(
            gate_bias_sb.shape == up_bias_sb.shape,
            f"expected gate and up biases to have the same shapes, got {gate_bias_sb.shape=}, {up_bias_sb.shape=}",
        )
    elif gate_bias_sb != None or up_bias_sb != None:
        kernel_assert(
            False,
            f"expected gate and up biases to be both None or both not None",
        )
    kernel_assert(TILE_H == TILE_H_, f"Expected same number of partitions in input and weight, got {TILE_H}, {TILE_H_}")
    kernel_assert(
        n_H512_tiles == n_H512_tiles_,
        f"Expected same number of H tiles in input and weight, got {n_H512_tiles}, {n_H512_tiles_}",
    )

    # Tiling strategies for T, I
    TILE_T = min(_psum_fmax * 2 // _q_width, T)  # I_4 * TILE_T <= psum_fmax * 2 for bf16 PSUM
    n_T256_tiles = div_ceil(T, TILE_T)
    n_total_I512_tiles = div_ceil(I_local, MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM)
    I_4, TILE_I = _q_width, nl.tile_size.pmax

    # Step 2: Allocate output buffers
    out_shape = (TILE_I, n_total_I512_tiles, T, I_4)
    out_quant_shape = (TILE_I, n_total_I512_tiles, T)
    out_sb = nl.ndarray(out_shape, dtype=activation_compute_dtype, buffer=nl.sbuf)
    out_quant_sb = nl.ndarray(out_quant_shape, dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    out_scale_sb = nl.ndarray(out_quant_shape, dtype=nl.uint8, buffer=nl.sbuf)

    """
    Step 3: Fused gate projection, projection clamping (optional), activation function.
    Step 3.1: Compute W_mxfp4/8 (stationary) @ input_mxfp8 (moving) via _matmul_mx_accumulate helper.
    TODO: consider changing loop order to T, H, I, 4_I
    """
    for tile_t in nl.sequential_range(n_T256_tiles):
        # T dim slicing, handling case when T tile < 256_T
        tile_T_offset = TILE_T * tile_t
        tile_T_actual = min(TILE_T, T - tile_T_offset)
        tile_T_slice = nl.ds(tile_T_offset, tile_T_actual)
        for tile_i in nl.sequential_range(n_total_I512_tiles):
            cur_tile_I_size = min(
                MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM, I_local - tile_i * MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
            )
            cur_I_pdim_sz = cur_tile_I_size // _q_width
            out_psum = nl.ndarray((TILE_I, I_4, TILE_T), dtype=nl.bfloat16, buffer=nl.psum)
            cur_I128_tile_sz = cur_tile_I_size // _q_width
            _matmul_mx_accumulate(
                out_psum=out_psum,
                weight_sb=gate_weight_sb,
                weight_scale_sb=gate_weight_scale_sb,
                input_quant_sb=input_quant_sb,
                input_scale_sb=input_scale_sb,
                tile_i=tile_i,
                tile_T_slice=tile_T_slice,
                tile_T_actual=tile_T_actual,
                n_H512_tiles=n_H512_tiles,
                cur_I128_tile_sz=cur_I128_tile_sz,
            )

            """
            Step 3.2: Accumulate bias during PSUM eviction.
            out_sb shape: [TILE_I, n_total_I512_tiles, T, I_4]
            out_psum shape: [TILE_I, I_4, TILE_T]
            gate_bias_sb shape: [TILE_I, n_total_I512_tiles, I_4]
            Use strided access pattern to reorder from [TILE_I, I_4, TILE_T] to [TILE_I, TILE_T, I_4].
            Use cur_I_pdim_sz for actual partitions.
            """
            if gate_bias_sb != None:
                nisa.tensor_tensor(
                    dst=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, :],
                    data1=out_psum.ap([[I_4 * TILE_T, cur_I_pdim_sz], [1, tile_T_actual], [TILE_T, I_4]]),
                    op=nl.add,
                    data2=gate_bias_sb.ap(
                        [[n_total_I512_tiles * I_4, cur_I_pdim_sz], [0, tile_T_actual], [1, I_4]], offset=tile_i * I_4
                    ),
                )
            else:
                nisa.tensor_copy(
                    dst=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, :],
                    src=out_psum.ap([[I_4 * TILE_T, cur_I_pdim_sz], [1, tile_T_actual], [TILE_T, I_4]]),
                )

            # Step 3.3: Clamp projection output to [clamp_lower_limit, clamp_upper_limit] (optional)
            _clamp_tensor(
                tensor=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, :],
                clamp_upper_limit=gate_clamp_upper_limit,
                clamp_lower_limit=gate_clamp_lower_limit,
            )

            # Step 3.4: Compute activation function
            if hidden_act_fn != None:
                nisa.activation(
                    dst=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, :],
                    data=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, :],
                    op=get_nl_act_fn_from_type(hidden_act_fn),
                )

    # Step 4: Fused up projection, projection clamp (optional), gate * up, MX quantization
    # Step 4.1: Compute W_mxfp4/8 (stationary) @ input_mxfp8 (moving)
    for tile_t in nl.sequential_range(n_T256_tiles):
        # T dim slicing, handling case when T tile < 256_T
        tile_T_offset = TILE_T * tile_t
        tile_T_actual = min(TILE_T, T - tile_T_offset)
        tile_T_slice = nl.ds(tile_T_offset, tile_T_actual)
        for tile_i in nl.sequential_range(n_total_I512_tiles):
            cur_tile_I_size = min(
                MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM, I_local - tile_i * MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
            )
            cur_I_pdim_sz = cur_tile_I_size // _q_width
            intermediate_tile_sb = nl.ndarray((TILE_I, 1, TILE_T, I_4), dtype=out_sb.dtype, buffer=nl.sbuf)
            out_psum = nl.ndarray((TILE_I, I_4, TILE_T), dtype=nl.bfloat16, buffer=nl.psum)
            cur_I128_tile_sz = cur_tile_I_size // _q_width
            _matmul_mx_accumulate(
                out_psum=out_psum,
                weight_sb=up_weight_sb,
                weight_scale_sb=up_weight_scale_sb,
                input_quant_sb=input_quant_sb,
                input_scale_sb=input_scale_sb,
                tile_i=tile_i,
                tile_T_slice=tile_T_slice,
                tile_T_actual=tile_T_actual,
                n_H512_tiles=n_H512_tiles,
                cur_I128_tile_sz=cur_I128_tile_sz,
            )

            """
            Step 4.2: Accumulate bias during PSUM eviction.
            intermediate_tile_sb shape: [TILE_I, 1, TILE_T, I_4]
            out_psum shape: [TILE_I, I_4, TILE_T]
            up_bias_sb shape: [TILE_I, n_total_I512_tiles, I_4]
            Use strided access pattern to reorder from [TILE_I, I_4, TILE_T] to [TILE_I, TILE_T, I_4].
            Use cur_I_pdim_sz for actual partitions.
            """
            if up_bias_sb != None:
                nisa.tensor_tensor(
                    dst=intermediate_tile_sb[:cur_I_pdim_sz, 0, :tile_T_actual, :],
                    data1=out_psum.ap([[I_4 * TILE_T, cur_I_pdim_sz], [1, tile_T_actual], [TILE_T, I_4]]),
                    op=nl.add,
                    data2=up_bias_sb.ap(
                        [[n_total_I512_tiles * I_4, cur_I_pdim_sz], [0, tile_T_actual], [1, I_4]], offset=tile_i * I_4
                    ),
                )
            else:
                nisa.tensor_copy(
                    dst=intermediate_tile_sb[:cur_I_pdim_sz, 0, :tile_T_actual, :],
                    src=out_psum.ap([[I_4 * TILE_T, cur_I_pdim_sz], [1, tile_T_actual], [TILE_T, I_4]]),
                )

            # Step 4.3: Clamp projection output to [clamp_lower_limit, clamp_upper_limit]
            _clamp_tensor(
                tensor=intermediate_tile_sb[:cur_I_pdim_sz, 0, :tile_T_actual, :],
                clamp_upper_limit=up_clamp_upper_limit,
                clamp_lower_limit=up_clamp_lower_limit,
            )

            # Step 4.4: Multiply completed up tile with corresponding gate tile
            nisa.tensor_tensor(
                dst=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, :],
                data1=out_sb[:cur_I_pdim_sz, tile_i, tile_T_slice, :],
                op=nl.multiply,
                data2=intermediate_tile_sb[:cur_I_pdim_sz, 0, :tile_T_actual, :],
            )

            # Step 4.5: MX quantize combined gate * up tile
            # Pad partition count to valid quantize_mx size {32, 64, 96, 128}.
            # Extra zero-padded partitions are harmless: downstream weight is zero-padded.
            qmx_I_pdim_sz = pad_to_valid_qmx_partitions(cur_I_pdim_sz)
            nisa.quantize_mx(
                src=out_sb[:qmx_I_pdim_sz, tile_i, tile_T_slice, :],
                dst=out_quant_sb[:qmx_I_pdim_sz, tile_i, tile_T_slice],
                dst_scale=out_scale_sb[:qmx_I_pdim_sz, tile_i, tile_T_slice],
            )

    return out_quant_sb, out_scale_sb


@nki.jit
def load_gate_up_weight_scale_bias(
    weight: nl.ndarray,
    scale: nl.ndarray,
    bias: Optional[nl.ndarray],
    expert_idx: int,
    gate_or_up_idx: int,
    H: int,
    n_I512_tiles_local: int,
    I_local: int,
    I_offset: int,
    I_local_padded: int = 0,
) -> tuple[nl.ndarray, nl.ndarray, Optional[nl.ndarray]]:
    """
    Load gate or up projection weight, scale, and bias (optional) for one expert using static DMA.

    When executed with LNC=2, weights and scales are sharded on I/512 tiles dimension (tile-based sharding).
    This ensures alignment with down_projection_mx which also uses tile-based I-sharding.

    Args:
        weight (nl.ndarray): [E_L, 128_H, 2, H/512, I], Gate or up projection weight tensor from HBM
            (fused gate/up weights), 4_H packed in x4 dtype.
        scale (nl.ndarray): [E_L, 16_H, 2, H/512, I], Gate or up projection MX scale tensor from HBM
            (fused gate/up scales), uint8 MX scales.
        bias (Optional[nl.ndarray]): [E_L, 128_I, 2, I/512, 4_I], Optional gate or up projection bias
            tensor from HBM (fused gate/up biases).
        expert_idx (int): Index of the current expert to load.
        gate_or_up_idx (int): Index to select gate (0) or up (1) projection from fused tensor.
        H (int): Hidden dimension size.
        n_I512_tiles_local (int): Number of I/512 tiles for this NC (may differ between NCs for odd tile counts).
        I_local (int): Local intermediate dimension size for this NC.
        I_offset (int): Starting I offset for this NC's tiles.
        I_local_padded (int): Padded I_local (nearest multiple of 8). If 0, defaults to I_local (no padding).

    Returns:
        weight_sb (nl.ndarray): [128_H, H/512, I_local_padded], Weight in SBUF (4_H packed in x4 dtype).
        scale_sb (nl.ndarray): [128_H, H/512, I_local_padded], Scales in SBUF (in leading 4P of each SBUF quadrant).
        bias_sb (Optional[nl.ndarray]): [128_I, n_I512_tiles_local, 4_I], Bias in SBUF (None when bias not provided).

    Notes:
        - Uses tile-based I-sharding to align with down_projection_mx
        - NC0 gets first n_I512_tiles_local tiles, NC1 gets the rest
        - Based on experiments, static DMA demonstrates better performance
    """

    # Calculate shapes / tiling
    I_buf = I_local_padded if I_local_padded > 0 else I_local
    kernel_assert(
        I_buf % MIN_MATMULT_MX_P_DIM == 0,
        f"Expected I_local (padded) divisible by {MIN_MATMULT_MX_P_DIM} for nc_matmul_mx even free-dim constraint, got {I_buf=}.",
    )
    needs_padding = I_buf > I_local
    pmax = nl.tile_size.pmax
    TILE_H, n_H512_tiles = pmax, H // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM
    TILE_I, I_4 = pmax, _q_width
    weight_sb_shape = (TILE_H, n_H512_tiles, I_buf)
    bias_sb_shape = (TILE_I, n_I512_tiles_local, I_4)
    is_bias = bias != None

    # Allocate buffers
    base_weight = weight.base_tensor
    weight_sb = nl.ndarray(weight_sb_shape, dtype=base_weight.dtype, buffer=nl.sbuf)
    scale_sb = nl.ndarray(weight_sb_shape, dtype=scale.dtype, buffer=nl.sbuf)
    bias_sb = nl.ndarray(bias_sb_shape, dtype=bias.dtype, buffer=nl.sbuf) if is_bias else None

    # Load weight: index expert and gate/up, then slice I dimension using tile-based offset
    # Shape: [E_L, 128_H, 2, H/512, I] -> [128_H, H/512, I_local] -> padded to [128_H, H/512, I_buf]
    weight_view = (
        TensorView(base_weight)
        .select(dim=0, index=expert_idx)
        .select(dim=1, index=gate_or_up_idx)
        .slice(dim=2, start=I_offset, end=I_offset + I_local)
    )
    if needs_padding:
        nisa.memset(dst=weight_sb[...], value=0, engine=nisa.gpsimd_engine)
        nisa.dma_copy(src=weight_view.get_view(), dst=weight_sb[:, :, :I_local])
    else:
        nisa.dma_copy(src=weight_view.get_view(), dst=weight_sb[...])
    weight_sb = weight_sb.view(weight.dtype)

    """
    Load scale: index expert and gate/up, then slice I dimension using tile-based offset.
    Shape: [E_L, 16_H, 2, H/512, I] -> [16_H, H/512, I_local]
    Scale layout: 16 partitions map to partitions [0-3, 32-35, 64-67, 96-99] in 128-partition buffer.
    """
    n_scale_partitions = TILE_H // _q_height
    n_quadrants_needed = div_ceil(n_scale_partitions, SCALE_P_ELEM_PER_QUADRANT)

    if needs_padding:
        nisa.memset(dst=scale_sb[...], value=0.0, engine=nisa.gpsimd_engine)

    for quadrant_idx in nl.affine_range(n_quadrants_needed):
        scale_view = (
            TensorView(scale)
            .select(dim=0, index=expert_idx)
            .slice(
                dim=0,
                start=SCALE_P_ELEM_PER_QUADRANT * quadrant_idx,
                end=SCALE_P_ELEM_PER_QUADRANT * (quadrant_idx + 1),
            )
            .select(dim=1, index=gate_or_up_idx)
            .slice(dim=2, start=I_offset, end=I_offset + I_local)
        )
        nisa.dma_copy(
            src=scale_view.get_view(),
            dst=scale_sb[nl.ds(SBUF_QUADRANT_SIZE * quadrant_idx, SCALE_P_ELEM_PER_QUADRANT), :, :I_local],
        )

    tile_offset = I_offset // MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM

    # Load bias: index expert and gate/up, then slice I/512 tiles based on tile ownership
    # Shape: [E_L, I_p, 2, I/512, 4_I] -> [I_p, n_I512_tiles_local, 4_I] -> padded to [128_I, n_I512_tiles_local, 4_I]
    if is_bias:
        I_p_bias_in_hbm = bias.shape[1]

        if I_p_bias_in_hbm < TILE_I:
            nisa.memset(dst=bias_sb[...], value=0.0, engine=nisa.gpsimd_engine)

        bias_view = (
            TensorView(bias)
            .select(dim=0, index=expert_idx)
            .select(dim=1, index=gate_or_up_idx)
            .slice(dim=1, start=tile_offset, end=tile_offset + n_I512_tiles_local)
        )
        nisa.dma_copy(
            src=bias_view.get_view(),
            dst=bias_sb[:I_p_bias_in_hbm, :, :],
        )

    return weight_sb, scale_sb, bias_sb


@nki.jit
def _clamp_tensor(
    tensor: nl.ndarray,
    clamp_upper_limit: Optional[float],
    clamp_lower_limit: Optional[float],
) -> None:
    """
    Apply optional clamping to a tensor in-place.

    Args:
        tensor (nl.ndarray): Tensor slice to clamp.
        clamp_upper_limit (Optional[float]): Upper clamp limit (None to skip).
        clamp_lower_limit (Optional[float]): Lower clamp limit (None to skip).
    """
    if clamp_upper_limit != None or clamp_lower_limit != None:
        nisa.tensor_scalar(
            dst=tensor,
            data=tensor,
            op0=nl.minimum if clamp_upper_limit != None else None,
            operand0=clamp_upper_limit,
            op1=nl.maximum if clamp_lower_limit != None else None,
            operand1=clamp_lower_limit,
        )


@nki.jit
def _matmul_mx_accumulate(
    out_psum: nl.ndarray,
    weight_sb: nl.ndarray,
    weight_scale_sb: nl.ndarray,
    input_quant_sb: nl.ndarray,
    input_scale_sb: nl.ndarray,
    tile_i: int,
    tile_T_slice,
    tile_T_actual: int,
    n_H512_tiles: int,
    cur_I128_tile_sz: int,
) -> None:
    """
    Perform MX matmul accumulation over H tiles and 4-I blocks.

    This helper extracts the common matmul loop used in both gate and up projections.

    Args:
        out_psum (nl.ndarray): [TILE_I, I_4, TILE_T], Output PSUM buffer.
        weight_sb (nl.ndarray): [128_H, H/512, I], Weight tensor in SBUF.
        weight_scale_sb (nl.ndarray): [128_H, H/512, I], Weight scale tensor in SBUF.
        input_quant_sb (nl.ndarray): [128_H, H/512, T], Quantized input in SBUF.
        input_scale_sb (nl.ndarray): [128_H, H/512, T], Input scale in SBUF.
        tile_i (int): Current I tile index.
        tile_T_slice: T dimension slice descriptor.
        tile_T_actual (int): Actual T tile size (may be < 256 for last tile).
        n_H512_tiles (int): Number of H/512 tiles.
        cur_I128_tile_sz (int): Current I tile size in partitions (I_size / 4).
    """
    for q_width_I_idx in nl.sequential_range(_q_width):
        weight_I_offset = tile_i * MAX_MATMULT_MX_UNPACKED_CONTRACT_DIM + q_width_I_idx * cur_I128_tile_sz
        weight_I_slice = nl.ds(weight_I_offset, cur_I128_tile_sz)
        for tile_h in nl.sequential_range(n_H512_tiles):
            nisa.nc_matmul_mx(
                dst=out_psum[:cur_I128_tile_sz, q_width_I_idx, :tile_T_actual],
                stationary=weight_sb[:, tile_h, weight_I_slice],
                moving=input_quant_sb[:, tile_h, tile_T_slice],
                stationary_scale=weight_scale_sb[:, tile_h, weight_I_slice],
                moving_scale=input_scale_sb[:, tile_h, tile_T_slice],
            )
```

## nkilib/core/subkernels/indexed_flatten.py

```python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Indexed flatten kernel for MoE blockwise matmul operations."""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil


@nki.jit
def indexed_flatten(
    input_tensor: nl.ndarray,
    f_len: int,
    output_len: int,
    row_offsets: nl.ndarray,
    row_offsets_start: Optional[nl.ndarray] = None,
    padding_val: int = -1,
) -> nl.ndarray:
    """
    Indexed flatten kernel for MoE blockwise matmul operations.

    For an input_tensor of shape [E, T] and a set of row_offsets, this kernel
    reshapes the input to [E, T//f_len, f_len] and writes each row's data into
    the output tensor at the specified block offsets. Out-of-bounds offsets are
    skipped via nisa.oob_mode.skip. Optimized for LNC2 execution with all-reduce
    max aggregation between NeuronCores. Best performance for T <= 10240 elements
    per row. Using T > 10240 may result in degraded performance compared to smaller
    configurations.

    Dimensions:
        E: Number of rows (experts) in input tensor
        T: Number of elements per row
        N: Number of row offsets provided
        f_len: Block size in free dimension for DMA copies

    Args:
        input_tensor (nl.ndarray): [E, T], Input tensor on HBM
        f_len (int): Number of elements in each DMA copy in the free dimension
        output_len (int): Length of the output array
        row_offsets (nl.ndarray): [N,], Block offsets for each row on HBM
        row_offsets_start (Optional[nl.ndarray]): Optional start index for row_offsets
        padding_val (int): Value to fill unwritten positions (default: -1)

    Returns:
        flattened_array (nl.ndarray): [output_len,], Flattened output array on shared HBM

    Notes:
        - Requires LNC2 (2 NeuronCores)
        - output_len must be divisible by P_MAX (128)
        - output_len must be divisible by f_len
        - T must be divisible by f_len
        - (T // f_len) must be divisible by 16 for DMAs to work
        - When row_offsets_start is None, N must equal E
        - When row_offsets_start is provided, N must be >= E

    Pseudocode:
        output = full(output_len, padding_val)
        output_blocks = output.reshape(output_len // f_len, f_len)
        input_reshaped = input_tensor.reshape(E, T // f_len, f_len)

        for e in range(E):
            block_offset = row_offsets[e]
            for p in range(T // f_len):
                out_block_idx = block_offset + p
                if out_block_idx < output_len // f_len:
                    output_blocks[out_block_idx] = input_reshaped[e, p]

        return output
    """
    index_dtype = input_tensor.dtype
    P_MAX = nl.tile_size.pmax
    E, T = input_tensor.shape
    N = row_offsets.shape[0]

    # Input validation
    kernel_assert(output_len % P_MAX == 0, f"output_len must be divisible by P_MAX ({P_MAX}), got {output_len}")
    kernel_assert(output_len % f_len == 0, f"output_len must be divisible by f_len, got {output_len=}, {f_len=}")
    kernel_assert(T % f_len == 0, f"T must be divisible by f_len, got {T=}, {f_len=}")
    kernel_assert((T // f_len) % 16 == 0, f"(T // f_len) must be divisible by 16, got {T // f_len}")

    # Handle row_offsets_start
    if row_offsets_start is None:
        kernel_assert(N == E, f"When row_offsets_start is None, N ({N}) must equal E ({E})")
        row_offsets_start_val = 0
    else:
        kernel_assert(N >= E, f"When row_offsets_start is provided, N ({N}) must be >= E ({E})")
        row_offsets_start_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=row_offsets_start_sb, src=row_offsets_start.reshape((1, 1)))
        row_offsets_start_val = row_offsets_start_sb

    num_shards = nl.num_programs(0)
    shard_id = nl.program_id(0)

    # Calculate rows per shard, handling odd E
    # Shard 0 gets ceiling(E/2), Shard 1 gets floor(E/2)
    E_per_shard_0 = div_ceil(E, num_shards)
    E_per_shard_1 = E // num_shards
    E_per_shard = E_per_shard_0 if shard_id == 0 else E_per_shard_1
    E_offset = 0 if shard_id == 0 else E_per_shard_0

    num_output_blocks = output_len // f_len
    partitions_per_row = T // f_len
    partition_tile_count = div_ceil(partitions_per_row, P_MAX)

    # Each NC has its own private HBM buffer for partial results
    flattened_array_partial = nl.ndarray((num_output_blocks, f_len), dtype=index_dtype, buffer=nl.private_hbm)

    """
    Tiling Strategy:
    - Input [E, T] is reshaped to [E, partitions_per_row, f_len]
    - Each NC processes E_per_shard rows (shard 0 gets ceiling, shard 1 gets floor)
    - Partitions are processed in tiles of P_MAX (128) partitions
    - Each tile writes to output at dynamic offset via scalar_offset
    - Output is accumulated via all-reduce max between NCs
    """

    # Initialize output with padding
    sbuf_init = nl.ndarray((P_MAX, output_len // P_MAX), dtype=index_dtype, buffer=nl.sbuf)
    nisa.memset(dst=sbuf_init, value=padding_val)
    nisa.dma_copy(dst=flattened_array_partial.reshape((P_MAX, output_len // P_MAX)), src=sbuf_init)

    input_tensor_reshape = input_tensor.reshape((E, partitions_per_row, f_len))
    row_offsets_2d = row_offsets.reshape((1, N))

    # Use the maximum E_per_shard for loop bounds (both NCs iterate same number of times)
    max_E_per_shard = E_per_shard_0

    """
    Load offsets for this shard.
    Use a large negative value for invalid offsets to ensure all indices are OOB and skipped.
    """
    INVALID_OFFSET_VALUE = -1000000
    row_offsets_local = nl.ndarray((1, max_E_per_shard), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=row_offsets_local, value=INVALID_OFFSET_VALUE)
    if E_per_shard > 0:
        if row_offsets_start is None:
            nisa.dma_copy(
                dst=row_offsets_local[0:1, 0:E_per_shard],
                src=row_offsets_2d[0:1, E_offset : E_offset + E_per_shard],
            )
        else:
            # Load row_offsets starting from row_offsets_start + E_offset using .ap() with scalar_offset
            nisa.dma_copy(
                dst=row_offsets_local[0:1, 0:E_per_shard],
                src=row_offsets_2d.ap(
                    pattern=[[N, 1], [1, E_per_shard]],
                    offset=E_offset,
                    scalar_offset=row_offsets_start_val,
                    indirect_dim=1,
                ),
            )

    for row_idx_local in nl.sequential_range(max_E_per_shard):
        row_offset_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=row_offset_sb, src=row_offsets_local[0:1, row_idx_local : row_idx_local + 1])

        # Clamp row_idx to valid range
        row_idx = min(row_idx_local + E_offset, E - 1)

        for partition_tile_idx in nl.sequential_range(partition_tile_count):
            partition_start = partition_tile_idx * P_MAX
            partition_count = min(P_MAX, partitions_per_row - partition_start)

            if partition_count > 0:
                input_tile = nl.ndarray((partition_count, f_len), dtype=index_dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=input_tile,
                    src=input_tensor_reshape[row_idx, partition_start : partition_start + partition_count, 0:f_len],
                )

                # Use nisa.oob_mode.skip to skip writes for out-of-bounds offsets
                nisa.dma_copy(
                    dst=flattened_array_partial.ap(
                        pattern=[[f_len, partition_count], [1, f_len]],
                        offset=partition_start * f_len,
                        scalar_offset=row_offset_sb,
                        indirect_dim=0,
                    ),
                    src=input_tile,
                    oob_mode=nisa.oob_mode.skip,
                )

    # All-reduce max between the two NCs
    reshaped_reload = flattened_array_partial.reshape((P_MAX, output_len // P_MAX))
    reshaped_reload_local = nl.ndarray((P_MAX, output_len // P_MAX), dtype=index_dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=reshaped_reload_local, src=reshaped_reload)

    reshaped_reload_remote = nl.ndarray((P_MAX, output_len // P_MAX), dtype=index_dtype, buffer=nl.sbuf)
    nisa.sendrecv(
        src=reshaped_reload_local,
        dst=reshaped_reload_remote,
        send_to_rank=(1 - shard_id),
        recv_from_rank=(1 - shard_id),
        pipe_id=0,
    )

    result_sb = nl.ndarray((P_MAX, output_len // P_MAX), dtype=index_dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=result_sb, data1=reshaped_reload_local, data2=reshaped_reload_remote, op=nl.maximum)

    flattened_array = nl.ndarray((output_len,), dtype=index_dtype, buffer=nl.shared_hbm)
    if shard_id == 0:
        nisa.dma_copy(dst=flattened_array.reshape((P_MAX, output_len // P_MAX)), src=result_sb)

    return flattened_array
```

## nkilib/core/qkv/qkv_tkg_mx_impl.py

```python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QKV TKG MXFP Projection Kernel for token generation with MX quantization."""

import math
from typing import Optional, Tuple

import nki.isa as nisa
import nki.language as nl

from ..mlp.mlp_tkg.mlp_tkg_utils import _layout_adapter_sb
from ..subkernels.rmsnorm_tkg import rmsnorm_tkg as _rmsnorm_tkg
from ..utils.allocator import SbufManager, sizeinbytes
from ..utils.common_types import NormType, QKVOutputLayout, QuantizationType
from ..utils.kernel_assert import kernel_assert
from ..utils.tiled_range import TiledRange
from .qkv_tkg_mx_utils import (
    QKV_TKG_MXFP_Config,
    QKV_TKG_MXFP_UserInput,
    _build_config,
    _validate_user_inputs,
)

# Tiling constants
I_BLOCK_SIZE = 4096  # I_TILE_SIZE * 8 for 8 available PSUM banks
I_TILE_SIZE = 512  # Maximum free dimension of a matmul instruction (one PSUM bank)
WEIGHT_LOAD_BLOCK_SIZE = 2048  # Number of rows to load per weight block
P_MAX = 128  # Partition dimension size (128)


def _qkv_tkg_mx_impl(
    hidden: nl.ndarray,
    weights_qtz_hbm: nl.ndarray,
    norm_w: Optional[nl.ndarray] = None,
    fused_add: bool = False,
    mlp_prev: Optional[nl.ndarray] = None,
    attn_prev: Optional[nl.ndarray] = None,
    d_head: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    num_q_heads: Optional[int] = None,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    eps: float = 1e-6,
    norm_type: NormType = NormType.RMS_NORM,
    quantization_type: QuantizationType = QuantizationType.MX,
    is_h_dim_4h_transposed: bool = False,
    weight_scales_hbm: Optional[nl.ndarray] = None,
    output_in_sbuf: bool = False,
    qkv_bias: Optional[nl.ndarray] = None,
    norm_bias: Optional[nl.ndarray] = None,
    hidden_actual: Optional[int] = None,
    sbm: Optional[SbufManager] = None,
) -> nl.ndarray:
    """
    QKV MXFP Projection Kernel for Token Generation
    MXFP specific additional assumptions (subset of features supported by non-quantized version):
        -> Assumes weights_hbm are already quantized and stored in nl.float8_e4m3fn_x4 dtype.
        -> Assumes weights_scales are passed in nl.uint8 dtype as well.
        -> is_h_dim_4h_transposed must be set to True.
        -> H = hidden.shape[2] must be divisible by 512.
        -> BxS must be divisible by 4.
        -> Input must be on HBM.

    This kernel computes the fused QKV projection operation:
        hidden' = norm(hidden + attn_prev + mlp_prev)  # optional fused add and norm
        output = hidden' @ qkv_w + qkv_bias
    typically used before the attention block in transformer models.

    This kernel is optimized for Token Generation (aka Decoding) use cases where
    batch_size * seqlen is small. This kernel only supports BxS <= 128.

    The kernel supports optional fused residual addition and normalization (RMSNorm/LayerNorm)
    to reduce HBM traffic and improve performance.

    Data Types:
        This kernel supports nl.float32, nl.float16, and nl.bfloat16 data types.

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension size
        I: Fused QKV output dimension ((Q + K + V) * N * D)
        N: Number of heads (for NBSd output layout)
        D: Head dimension size (for NBSd output layout)

    Args:
        hidden (nl.ndarray):
            Input hidden states tensor in HBM or SBUF.
            Shape:
                [B, S, H]         when in HBM
                [H0=128, BxS, H1] when in SBUF
        weights_qtz_hbm (nl.ndarray):
            QKV projection weight tensor in HBM, quantized offline.
            Dtype: nl.float8_e4m3fn_x4
            Shape: [H_packed, I], can be viewed as [H0, H1_packed, I].
            Note: Here H_packed = H // 4, where H = hidden.shape[2].
        norm_w (nl.ndarray, optional):
            Normalization weight tensor in HBM. Required when norm_type is RMS_NORM or LAYER_NORM.
            Shape:    [1, H]
        fused_add (bool):
            Enable fused residual addition (hidden + attn_prev + mlp_prev). Default: False.
        mlp_prev (nl.ndarray, optional):
            Previous MLP residual tensor in HBM. Required when fused_add is True.
            Shape:    [B, S, H]
        attn_prev (nl.ndarray, optional):
            Previous attention residual tensor in HBM. Required when fused_add is True.
            Shape:    [B, S, H]
        d_head (int, optional):
            Head dimension size D. Required for NBSd layout.
        num_q_heads : Optional[int], default=None
            Number of query heads
        num_kv_heads : Optional[int], default=None
            Number of key/value heads
        output_layout (QKVOutputLayout):
            Output tensor layout format. BSD: [B, S, I] or NBSd: [N, B, S, D]. Default: QKVOutputLayout.BSD.
        eps (float):
            Epsilon value to maintain numerical stability in normalization. Default: 1e-6.
        norm_type (NormType):
            Type of normalization to apply (NO_NORM, RMS_NORM, or LAYER_NORM). Default: NormType.RMS_NORM.
        quantization_type (QuantizationType):
            Must be QuantizationType.MX.
        is_h_dim_4h_transposed: bool, default=False
            Whether the H-dim (in input and gamma) has been pre-transposed by 4 (only applicable with MX Quantization).
            If is_h_dim_4h_transposed = False,
                * input has typical shape [B, S, H], viewed as [B, S, H//512, 128_H, 4_H].
            If is_h_dim_4h_transposed = True,
                * input has shape [B, S, H] but is pre-shuffled from
                  [B, S, H//512, 128_H, 4_H] -> [B, S, 4_H, H//512, 128_H] and flattened to [B, S, H].
                * IMPORTANT: H-dim in both input and gamma weights (for RMSNorm) must be pre-shuffled.
                    * For input, this is achieved by offline pre-shuffling weights of upstream projection (in real model).
                    * For gamma, this is achieved by offline pre-shuffling of gamma tensor.
                Purpose: More efficent for obtaining the required swizzled layout for quantize_mx instruction.
        weight_scales_hbm (nl.ndarray):
            QKV weight quantization scales for MXFP in HBM.
            dtype: uint8
            Shape: [H // 32, I] == [H_packed // 8, I]
            Note: Since weights_qtz_hbm is already quantized, weight scales are 8x times smaller, not 32x.
        output_in_sbuf (bool):
            If True, output is kept in SBUF; otherwise stored to HBM. Default: False.
            Only supports single I-block when True.
        qkv_bias (nl.ndarray, optional):
            Bias tensor in HBM for QKV projection.
            Shape:    [1, I]
        norm_bias (nl.ndarray, optional):
            LayerNorm beta parameter tensor in HBM. Required when norm_type is LAYER_NORM.
            Shape:    [1, H]
        hidden_actual (int, optional):
            Actual hidden dimension for padded input tensors. If specified, normalization
            uses this value instead of H for mean calculation.
        sbm (SbufManager, optional):
            Instance of SbufManager responsible for handling SBUF allocation.
            If None, auto-allocation manager is created.

    Returns:
        output (nl.ndarray | Tuple[nl.ndarray, nl.ndarray]):
            QKV projection output tensor. The tensor can reside in either SBUF or HBM.
            Shape:    [B, S, I] for BSD layout, [N, B, S, D] for NBSd layout.
            When fused_add is True, returns tuple (output, fused_hidden) where
            fused_hidden is the result of the fused residual addition.

    Notes:
        - H must be divisible by 128 (nl.tile_size.pmax).
        - H1 (H//128) must be divisible by number of shards for multi-core execution.
        - output_in_sbuf only supports single I-block (I < 4096).

    Pseudocode:
        # Step 1: Load input and optionally apply RMSNorm
        if norm_type == RMS_NORM:
            hidden_sb = rmsnorm(hidden, gamma, eps)
        else:
            hidden_sb = dma_transpose(hidden)  # [B*S, H] -> [H0, B*S, H1]

        # Step 2: Quantize input for MXFP
        hidden_swizzled = layout_adapter_sb(hidden_sb)  # [H0, B*S, H1] -> [H0, H1_packed, B*S, 4]
        hidden_qtz, hidden_scale = quantize_mx(hidden_swizzled)

        # Step 3: MXFP Projection with tiled matmul
        for i_block in range(NUM_I_BLOCKS):
            for h_block in range(NUM_H_BLOCKS):
                weights_qtz = load_weights(h_block, i_block)
                weight_scales = load_weight_scales(h_block, i_block)
                psum += nc_matmul_mx(hidden_qtz, weights_qtz, hidden_scale, weight_scales)
            output[i_block] = psum + bias  # if bias enabled
            if num_shards > 1:
                output[i_block] = sendrecv_reduce(output[i_block])
    """

    # Build user inputs and validate
    user_inputs = QKV_TKG_MXFP_UserInput(
        hidden=hidden,
        weights_qtz_hbm=weights_qtz_hbm,
        norm_w=norm_w,
        fused_add=fused_add,
        mlp_prev=mlp_prev,
        attn_prev=attn_prev,
        d_head=d_head,
        num_kv_heads=num_kv_heads,
        num_q_heads=num_q_heads,
        output_layout=output_layout,
        eps=eps,
        norm_type=norm_type,
        quantization_type=quantization_type,
        is_h_dim_4h_transposed=is_h_dim_4h_transposed,
        weight_scales_hbm=weight_scales_hbm,
        output_in_sbuf=output_in_sbuf,
        qkv_bias=qkv_bias,
        norm_bias=norm_bias,
        hidden_actual=hidden_actual,
        sbm=sbm,
    )
    _validate_user_inputs(user_inputs)

    # Build config
    cfg = _build_config(user_inputs)
    B, S, H = hidden.shape
    BxS = B * S
    H0 = P_MAX
    H1 = H // H0
    if hidden_actual is None:
        hidden_actual = H

    hidden_sb = nl.ndarray((H0, BxS, H1), dtype=cfg.hidden_orig_dtype, buffer=nl.sbuf)
    """
    High-Level Layout Path (ignoring sharding):
        HBM input               [BxS, H]
     -> dma_transpose to        [H0, BxS, H1] (+rms_norm), H1 is viewed as outer-dim now.
     -> view as                 [H0, BxS, 4_H, H1_packed] (is_h_dim_4h_transposed=True needed for this 4_H view)
     -> free-dim tensor copy to [H0, H1_packed, BxS, 4_H]
     -> quantize_mx to          [H0, H1_packed, BxS]
     -> ideal for matmul_mx
    """

    # Step 1: Load Input and optionally apply RMS_NORM
    """
    The key is to view H as H1 * H0 with H1 being outer-dimension, and use dma_transpose
    (in both NO_NORM or RMS_NORM). Along with is_h_dim_4h_transposed, this will allow
    efficient re-layout for quantize_mx later.
    """
    hidden_sb = _load_hidden_and_apply_rms_norm(
        hidden_hbm=hidden,
        output_sb=hidden_sb,
        cfg=cfg,
        gamma_hbm=norm_w,
        eps=eps,
        hidden_actual=hidden_actual,
    )

    # Step 2: Quantize_MX input
    # _quantize_mx_input(..) will re-layout input for quantization and return quantized result.
    # hidden_qtz_sb shape: [H0, H1_packed_shard, BxS]
    hidden_qtz_sb, hidden_scales_sb = _quantize_mx_input(hidden_sb=hidden_sb, cfg=cfg)

    # Step 3: MXFP Projection
    output_hbm = _qkv_tkg_projection_mxfp(
        hidden_qtz_sb=hidden_qtz_sb,
        hidden_scales_sb=hidden_scales_sb,
        weights_qtz_hbm=weights_qtz_hbm,
        weight_scales_hbm=weight_scales_hbm,
        cfg=cfg,
        bias_hbm=qkv_bias,
    )

    return output_hbm.reshape((cfg.B, cfg.S, cfg.I))


def _load_hidden_and_apply_rms_norm(
    hidden_hbm: nl.ndarray,
    output_sb: nl.ndarray,
    cfg: QKV_TKG_MXFP_Config,
    gamma_hbm: Optional[nl.ndarray],
    eps: Optional[float] = 1e-6,
    hidden_actual: Optional[int] = None,
):
    """
    Loads the input to hidden_sb and (optionally) applies RMSNorm.

    Args:
        hidden_hbm (nl.ndarray):
            Input hidden states in HBM.
            Shape: [B, S, H].

        output_sb (nl.ndarray):
            Input hidden states in SBUF (destination).
            Shape: [H0, BxS, H1]

        gamma_hbm (nl.ndarray):
            Normalization weight tensor in HBM. Required when norm_type is RMS_NORM or LAYER_NORM.
            Shape:    [1, H]

        cfg (QKV_TKG_MXFP_Config): QKV TKG MXFP configuration.

        hidden_actual (int, optional): Non-padded H for RMSNorm.


    Returns:
        Returns the result in output_sb, with H1 being the outer-dimension.
    """
    B, S, H = hidden_hbm.shape
    BxS = B * S
    H0 = P_MAX
    H1 = H // H0

    # Note: We cannot LNC2 shard this strided dma_transpose.
    if cfg.fused_norm_type == NormType.NO_NORM:
        # The key is to view H as H1 * H0 with H1 being outer-dimension.
        hidden_for_transpose = hidden_hbm.reshape((BxS * H1, H0)).reshape((BxS * H1, 1, 1, H0))
        hidden_sb_flat = output_sb.reshape((H0, BxS * H1)).reshape((H0, 1, 1, BxS * H1))
        nisa.dma_transpose(dst=hidden_sb_flat, src=hidden_for_transpose)
    else:  # cfg.fused_norm_type == NormType.RMS_NORM:
        # In case of RMSNorm we use hidden_dim_tp=True to indicate H1 needs to be viewed as the outer-dimension.
        output_sb = _rmsnorm_tkg(
            input=hidden_hbm,
            gamma=gamma_hbm,
            output=output_sb,
            eps=eps,
            hidden_actual=hidden_actual,
            hidden_dim_tp=True,  # IMPORTANT
            single_core_forced=True,
        )

    return output_sb


def _quantize_mx_input(
    hidden_sb: nl.ndarray,
    cfg: QKV_TKG_MXFP_Config,
) -> Tuple[nl.ndarray, nl.ndarray]:
    """
    Quantize hidden states for MXFP matmul.

    Args:
        hidden_sb (nl.ndarray):
            Input hidden states tensor in SBUF.
            Shape:
                [H0, BxS, H1] where H0=128, and H1 = H // H0
            Indexing: H is viewed as H1*H0 with H1 being the outer dimension.
                    -> Input was loaded with transpose.
        cfg (QKV_TKG_MXFP_Config): Kernel configuration.

    Note on is_h_dim_4h_transposed:
        This function assumes is_h_dim_4h_transposed=True.
        If swizzled, input has shape [B, S, H] but is pre-shuffled from
        [B, S, H//512, 128_H, 4_H] -> [B, S, 4_H, H//512, 128_H] and flattened to [B, S, H].

        Post dma_transpose, hidden_sb can be viewed as:
            [H0, BxS, H1] = [H0, BxS, 4_H * H // 512]
        This assumption allows for more efficient re-layout for quantize_mx.

    Returns:
        hidden_qtz_sb (nl.ndarray): [H0, H1_packed_shard, BxS], where H1_packed = H // 512, with nl.float8_e4m3fn_x4 dtype.
        hidden_scales_sb (nl.ndarray): [H0, H1_packed_shard, BxS], where H1_packed = H // 512, with nl.uint8 dtype.
    """
    H0, BxS, H1 = hidden_sb.shape
    H1_packed = H1 // 4
    H1_packed_shard = H1_packed // cfg.num_shards

    # Obtain needed layout for quantize_mx
    if cfg.is_h_dim_4h_transposed:
        """
        Pre-shuffling of H assumption (4_H being at front pre dma_transpose) allows us to
        obtain H layout necessary for quantization efficiently.
        START layout: [H0, BxS, H1] viewed as [H0, BxS, 4_H * H1_packed]
        GOAL layout for quantize_mx: [H0, H1_packed, BxS, 4_H]
        This can be achieved efficiently using free-dimension tensor_copy transpose
        in "_layout_adapter_sb" function.
        """
        hidden_swizzled_sb = _layout_adapter_sb(src=hidden_sb, n_prgs=cfg.num_shards, prg_id=cfg.shard_id)
        # Shape: [H0, H1_packed_shard, BxS, 4_H] - ready for quantize_mx
    else:
        kernel_assert(False, "[QKV TKG MXFP] is_dram_H_shuffled_with_4H_at_front=False is not implemented.")

    # Apply quantize_mx (_layout_adapter_sb returned sharded tensor)
    hidden_qtz_sb = nl.ndarray((H0, H1_packed_shard, BxS), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    hidden_scales_sb = nl.ndarray((H0, H1_packed_shard, BxS), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(dst=hidden_qtz_sb, src=hidden_swizzled_sb, dst_scale=hidden_scales_sb)

    return hidden_qtz_sb, hidden_scales_sb


def _qkv_tkg_projection_mxfp(
    hidden_qtz_sb: nl.ndarray,
    hidden_scales_sb: nl.ndarray,
    weights_qtz_hbm: nl.ndarray,
    weight_scales_hbm: nl.ndarray,
    cfg: QKV_TKG_MXFP_Config,
    bias_hbm: Optional[nl.ndarray] = None,
) -> nl.ndarray:
    """
    QKV MXFP Projection:
        Computes: hidden_qtz_sb @ weights_qtz_hbm.

    Note: This version differs from the current qkv_tkg projection in a few ways:
        (A) hidden_qtz_sb (SBUF) is assumed to be in [H0, H1, BxS] layout (with H1 being the outer dimension in H).
            * This differs from the original [H0, BxS, H1] layout, which won't work with MXFP.
        (B) No Column Tiling: MXFP version cannot have column tiling support (hardware incompatibility)

    Note: Here H_packed stands for the original H // 4.
    Args:
        hidden_qtz_sb (nl.ndarray):
            Input hidden states tensor in SBUF (already quantized).
            Dtype: nl.float8_e4m3fn_x4
            Shape:
                [H0, H1_packed_shard, BxS] where H0=128, and H1_packed_shard = H_packed_shard // H0.
                 e.g., H1_packed = H // 512 (or H_packed // 128).
            Indexing: h = h1*H0 + h0, i.e., H_packed is viewed as H1_packed*H0 with H1_packed being the outer dimension.
                Note: This means HBM->SBUF load requires a transpose.
                Torch equivalent of going from [H, BxS] -> [H0, H1, BxS] would be:
                    hidden_sb = input.reshape(H1, H0, BxS).permute(1, 0, 2)

        hidden_scales_sb (nl.ndarray):
            Input quantization scales for MXFP in SBUF.
            Dtype:  uint8
            Shape: [H0, H1_packed_shard, BxS],
            Note: Same indexing assumptions as hidden_qtz_sb.

        weights_qtz_hbm (nl.ndarray):
            QKV projection weight tensor in HBM.
            Dtype: nl.float8_e4m3fn_x4
            Shape: [H_packed, I], can be viewed as [H0, H1_packed, I].

        weight_scales_hbm (nl.ndarray):
            QKV weight quantization scales for MXFP in HBM.
            dtype: uint8
            Shape: [H // 32, I] == [H_packed // 8, I]
            Note: Since weights_qtz_hbm is already quantized, weight scales are 8x times smaller, not 32x.

        cfg (QKV_TKG_MXFP_Config): Kernel configuration (used for sharding info and dtype).

        bias_hbm (Optional[nl.ndarray]): [1, I], Optional bias tensor on HBM.

    Returns:
        output_hbm (nl.ndarray): [BxS, I], QKV projection output in HBM.
    """

    # Get sharding info from cfg
    num_shards = cfg.num_shards
    shard_id = cfg.shard_id
    hidden_orig_dtype = cfg.hidden_orig_dtype

    # We are deriving dimensions directly from already quantized tensors, hence shapes are 4x smaller already.
    H0, H1_packed_shard, BxS = hidden_qtz_sb.shape
    H_packed_shard = H0 * H1_packed_shard
    H_packed, I = weights_qtz_hbm.shape
    # hidden_qtz comes in LNC2 sharded, weights do not. We shard weights at load time with shard_id.
    # This is because we load weights with strided dma_copy pattern which cannot be done on pre-sliced tensor.

    weight_qtz_dtype = weights_qtz_hbm.dtype
    # Weight and weight scales asserts are done at the top-level of the kernel.
    kernel_assert(
        hidden_qtz_sb.dtype == nl.float8_e4m3fn_x4,
        f"[QKV TKG MXFP Kernel] _qkv_tkg_projection_mxfp(...) function expects hidden_qtz_sb.dtype == nl.float8_e4m3fn_x4, but got {hidden_qtz_sb.dtype}.",
    )
    kernel_assert(
        hidden_scales_sb.dtype == nl.uint8,
        f"[QKV TKG MXFP Kernel] _qkv_tkg_projection_mxfp(...) function expects hidden_scales_sb.dtype == nl.uint8, but got {hidden_scales_sb.dtype}.",
    )

    # Calculate MXFP tiling constants
    NUM_WEIGHT_LOAD_BLOCKS = math.ceil(H_packed_shard / WEIGHT_LOAD_BLOCK_SIZE)
    # NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK is the same constant as in non-MXFP kernel, however we are
    # processing 4x tiles per WEIGHT_LOAD_BLOCK. WEIGHT_LOAD_BLOCK is number of rows of H we
    # load/process at once, e.g., load only [2048, I] and do the compute.
    NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK = math.ceil(WEIGHT_LOAD_BLOCK_SIZE / P_MAX)
    NUM_I_BLOCKS = math.ceil(I / I_BLOCK_SIZE)

    # Allocate output in HBM
    output_hbm = nl.ndarray((BxS, I), dtype=hidden_orig_dtype, buffer=nl.shared_hbm)

    if cfg.add_bias:
        # Load Bias (1, I) to SBUF as (1, I), and broadcast it to (128, I) using stream_shuffle.
        bias_sb = _load_and_broadcast_bias(bias_hbm=bias_hbm, cfg=cfg)

    # QKV Projection loop - process each I_BLOCK_SIZE=4096 block (independent columns accumulated separately)
    for i_block in TiledRange(I, I_BLOCK_SIZE):
        # Allocate qkv_out_sb to store results of current I_BLOCK chunk
        qkv_out_sb = nl.ndarray((BxS, i_block.size), dtype=hidden_orig_dtype, buffer=nl.sbuf)

        """
        SBUF can run out-of-space in unallocated kernels. Add "list block" dimensions here,
        and/or make the kernel allocated. Also for now, SBUF space is approximate, use hardware
        specific SBUF space. None of this should be necessary in unallocated kernels, but it
        seems to be going out-of-space.
        """

        # Allocate weight buffer with double-buffering to overlap DMA with compute (if space permits)
        weight_tile_size = H0 * NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK * i_block.size * sizeinbytes(weight_qtz_dtype)
        weight_and_weight_scales_tile_size = 2 * weight_tile_size
        # Note: Quick heuristic for now, to be improved later. Since this is unallocated kernel we cannot know exact space taken.
        WEIGHT_MULTI_BUFFER_THERSHOLD_HERUISTIC = 16 * 1024 * 1024
        NUM_W_BUFFERS = 2 if weight_and_weight_scales_tile_size * 2 <= WEIGHT_MULTI_BUFFER_THERSHOLD_HERUISTIC else 1

        """
        If WEIGHT_LOAD_BLOCK_SIZE were maximum of H_packed, then weight_qtz_sb shape would be:
        (H0, NUM_W_BUFFERS, H1_packed = H // 512, i_block_sz).
        We use fixed block size, but since H in weights is packed, we process fewer H_BLOCKS
        than in non-quant version.
        """
        weights_qtz_sb = nl.ndarray(
            (H0, NUM_W_BUFFERS, NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK, i_block.size),
            dtype=nl.float8_e4m3fn_x4,
            buffer=nl.sbuf,
        )

        weight_scales_sb = nl.ndarray(
            (H0, NUM_W_BUFFERS, NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK, i_block.size),
            dtype=nl.uint8,
            buffer=nl.sbuf,
        )

        # Allocate PSUM banks for accumulation (num_i_tiles_per_i_block <= 8)
        mm_result_psum = []
        for i_tile in TiledRange(i_block.size, I_TILE_SIZE):
            psum_tile = nl.ndarray(
                (P_MAX, i_tile.size),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            mm_result_psum.append(psum_tile)

        # Process WEIGHT_LOAD_BLOCK_SIZE at a time, e.g., load [WEIGHT_LOAD_BLOCK_SIZE, I] and do the compute.
        for h_weight_block in TiledRange(H_packed_shard, WEIGHT_LOAD_BLOCK_SIZE):
            # TODO: Remove this once below loops are changed to TiledRange.
            num_128_tiles_in_current_weight_load_block = math.ceil(h_weight_block.size / P_MAX)

            # Buffer index for double-buffering (cycles 0, 1, 0, 1, ...)
            weight_buffer_idx = h_weight_block.index % NUM_W_BUFFERS

            """
            Load weight tile with single DMA using access patterns.
            Contiguous H sharding: shard 0 gets H[0:H/2], shard 1 gets H[H/2:H].
            HBM source pattern: weights_qtz_hbm is [H_packed, I].
            Offset into this shard's contiguous H portion.
            """
            h_shard_base_offset = shard_id * H_packed_shard * I
            weights_hbm_load_pattern = [
                [I, P_MAX],
                [P_MAX * I, num_128_tiles_in_current_weight_load_block],
                [1, i_block.size],
            ]
            weights_hbm_load_offset = h_shard_base_offset + h_weight_block.index * I + i_block.start_offset

            # SBUF dest pattern: weights_qtz_sb is [H0, NUM_W_BUFFERS, NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK, i_block_sz]
            weights_sb_load_pattern = [
                [NUM_W_BUFFERS * NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK * i_block.size, P_MAX],
                [i_block.size, num_128_tiles_in_current_weight_load_block],
                [1, i_block.size],
            ]
            weights_sb_load_offset = weight_buffer_idx * NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK * i_block.size

            # Load weights
            nisa.dma_copy(
                dst=weights_qtz_sb.ap(pattern=weights_sb_load_pattern, offset=weights_sb_load_offset),
                src=weights_qtz_hbm.ap(pattern=weights_hbm_load_pattern, offset=weights_hbm_load_offset),
            )

            """
            Load packed weight scales.
            MX Scale layout constants:
            Quadrant placement: HBM has 16 contiguous scale rows per H512 tile.
            SBUF needs them spread across 4 quadrants (4 rows each at offsets 0, 32, 64, 96).
            """
            SCALE_GROUP_SIZE = 32  # One scale per 32 H elements
            SCALES_PER_H512_TILE = 512 // SCALE_GROUP_SIZE  # = 16 scale rows per H512 tile in HBM
            SBUF_QUADRANT_SIZE = 32
            NUM_QUADRANTS = H0 // SBUF_QUADRANT_SIZE  # = 4
            SCALES_PER_QUADRANT = SCALES_PER_H512_TILE // NUM_QUADRANTS  # = 4

            # weight_scales_hbm shape: [H // 32, I] = [H_packed // 8, I]
            # Contiguous sharding: shard's scale rows start at shard_id * (H_packed_shard // 8)
            H_packed_shard_scale_rows = H_packed_shard // 8  # Number of scale rows for this shard
            scale_shard_base_offset = shard_id * H_packed_shard_scale_rows * I

            # num_128_tiles_in_current_weight_load_block iterations
            # for h_tile_idx_in_block in range(num_128_tiles_in_current_weight_load_block):
            for h_tile_in_block in TiledRange(h_weight_block.size, P_MAX):
                h_tile_idx_in_h = h_weight_block.index * NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK + h_tile_in_block.index

                for quad_idx in range(NUM_QUADRANTS):
                    # Local offset within this shard's scale rows
                    local_hbm_row_offset = (
                        h_tile_idx_in_h * SCALES_PER_H512_TILE + quad_idx * SCALES_PER_QUADRANT
                    ) * I + i_block.start_offset
                    hbm_row_offset = scale_shard_base_offset + local_hbm_row_offset

                    nisa.dma_copy(
                        dst=weight_scales_sb[
                            quad_idx * SBUF_QUADRANT_SIZE : quad_idx * SBUF_QUADRANT_SIZE + SCALES_PER_QUADRANT,
                            weight_buffer_idx,
                            h_tile_in_block.index,
                            : i_block.size,
                        ],
                        src=weight_scales_hbm.ap(
                            pattern=[[I, SCALES_PER_QUADRANT], [1, i_block.size]], offset=hbm_row_offset, dtype=nl.uint8
                        ),
                    )

            # Matmul for current weight load
            # num_128_tiles_in_current_weight_load_block iterations
            for h_tile_in_block in TiledRange(h_weight_block.size, P_MAX):
                h_tile_idx_in_h = h_weight_block.index * NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK + h_tile_in_block.index

                for i_tile in TiledRange(i_block.size, I_TILE_SIZE):
                    nisa.nc_matmul_mx(
                        dst=mm_result_psum[i_tile.index][0:BxS, 0 : i_tile.size],
                        stationary=hidden_qtz_sb[0:H0, h_tile_idx_in_h, 0:BxS],
                        moving=weights_qtz_sb[
                            0:H0,
                            weight_buffer_idx,
                            h_tile_in_block.index,
                            i_tile.start_offset : i_tile.start_offset + i_tile.size,
                        ],
                        stationary_scale=hidden_scales_sb[0:H0, h_tile_idx_in_h, 0:BxS],
                        moving_scale=weight_scales_sb[
                            0:H0,
                            weight_buffer_idx,
                            h_tile_in_block.index,
                            i_tile.start_offset : i_tile.start_offset + i_tile.size,
                        ],
                    )

        # Copy PSUM results to SBUF.
        for i_tile in TiledRange(i_block.size, I_TILE_SIZE):
            # So we don't double-add bias.
            if cfg.add_bias and cfg.shard_id == 0:
                nisa.tensor_tensor(
                    dst=qkv_out_sb[:BxS, i_tile.start_offset : i_tile.start_offset + i_tile.size],
                    data1=mm_result_psum[i_tile.index][:BxS, 0 : i_tile.size],
                    data2=bias_sb[
                        :BxS,
                        i_block.start_offset + i_tile.start_offset : i_block.start_offset
                        + i_tile.start_offset
                        + i_tile.size,
                    ],
                    op=nl.add,
                )
            else:
                nisa.tensor_copy(
                    dst=qkv_out_sb[:BxS, i_tile.start_offset : i_tile.start_offset + i_tile.size],
                    src=mm_result_psum[i_tile.index][:BxS, 0 : i_tile.size],
                )

        # Cross-core reduction via sendrecv when LNC > 1
        # Each core has partial sum from its H shard, need to sum across cores
        if num_shards > 1:
            qkv_recv_sb = nl.ndarray((BxS, i_block.size), dtype=hidden_orig_dtype, buffer=nl.sbuf)
            other_core = 1 - shard_id
            nisa.sendrecv(
                src=qkv_out_sb,
                dst=qkv_recv_sb,
                send_to_rank=other_core,
                recv_from_rank=other_core,
                pipe_id=0,
            )
            nisa.tensor_tensor(dst=qkv_out_sb, data1=qkv_out_sb, data2=qkv_recv_sb, op=nl.add)

        # Store to HBM
        nisa.dma_copy(
            dst=output_hbm[:BxS, i_block.start_offset : i_block.start_offset + i_block.size],
            src=qkv_out_sb[:BxS, 0 : i_block.size],
        )

    return output_hbm


def _load_and_broadcast_bias(
    bias_hbm: nl.ndarray,
    cfg: QKV_TKG_MXFP_Config,
) -> nl.ndarray:
    """
    Load bias and broadcast to partition dimension.

    Loads bias with shape [1, I] to SBUF and broadcasts it to [nl.tile_size.pmax, I]
    using stream_shuffle.

    Args:
        bias_hbm (nl.ndarray): [1, I], Bias tensor in HBM.
        cfg (QKV_TKG_MXFP_Config): Kernel configuration.

    Returns:
        nl.ndarray: [nl.tile_size.pmax, I], Broadcasted bias tensor in SBUF.
    """
    _, I = bias_hbm.shape

    # Load Bias (1, I) to SBUF as (1, I), and broadcast it to (128, I) using stream_shuffle.
    bias_sb = nl.ndarray((nl.tile_size.pmax, I), dtype=cfg.hidden_orig_dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=bias_sb[0:1, 0:I],
        src=bias_hbm[0:1, 0:I],
    )

    # Stream Shuffle works on 32 partitions only, apply it nl.tile_size.pmax // 32 = 4 times.
    MAX_STREAM_SHUFFLE_PARTITIONS = 32
    NUM_BROADCASTS = nl.tile_size.pmax // MAX_STREAM_SHUFFLE_PARTITIONS
    for broadcast_idx in nl.affine_range(NUM_BROADCASTS):
        nisa.nc_stream_shuffle(
            dst=bias_sb[
                nl.ds(broadcast_idx * MAX_STREAM_SHUFFLE_PARTITIONS, MAX_STREAM_SHUFFLE_PARTITIONS),
                0:I,
            ],
            src=bias_sb[0:1, 0:I],
            shuffle_mask=[0] * MAX_STREAM_SHUFFLE_PARTITIONS,
        )
    return bias_sb
```

## nkilib/experimental/transformer/transformer_tkg.py

```python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer forward pass optimized for token generation (TKG), composing attention_block_tkg and MLP kernels."""

from typing import List, Optional

import nki.collectives as nccl
import nki.isa as nisa
import nki.language as nl

from ...core.mlp.mlp import mlp
from ...core.utils.allocator import BufferManager, Logger
from ...core.utils.common_types import ActFnType, NormType, QuantizationType
from ...core.utils.kernel_helpers import get_verified_program_sharding_info
from ...core.utils.tensor_view import TensorView
from .attention_block_tkg import attention_block_tkg

_SBM_SIZE_BYTES = 200 * 1024  # Buffer manager size in bytes

# ==================== Helper Functions (module-level for compiler compatibility) ====================


def _load_input_to_sbuf(dst_sb, src_hbm, BxS: int, H0: int, H1: int, H1_shard: int, n_prgs: int):
    """Load [B, S_tkg, H] HBM tensor to [H0, BxS*H1] SBUF layout."""
    src_view = TensorView(src_hbm.reshape((BxS, H0 * H1))).rearrange(
        ('bs', ('lnc', 'h0', 'h1')), ('h0', 'bs', 'lnc', 'h1'), {'lnc': n_prgs, 'h0': H0}
    )
    dst_reshaped = dst_sb.reshape((H0, BxS, n_prgs, H1_shard))
    for lnc_idx in nl.static_range(n_prgs):
        nisa.dma_copy(
            src=src_view.slice(dim=2, start=lnc_idx, end=lnc_idx + 1).get_view(),
            dst=dst_reshaped[:, :, lnc_idx : lnc_idx + 1, :],
        )


def _store_output_to_hbm(out_hbm, in_sb, BxS: int, H0: int, H1: int, H1_shard: int, n_prgs: int):
    """Store [H0, BxS*H1] SBUF tensor to [B, S_tkg, H] HBM layout."""
    src_reshaped = in_sb.reshape((H0, BxS, n_prgs, H1_shard))
    dst_view = TensorView(out_hbm.reshape((BxS, H0 * H1))).rearrange(
        ('bs', ('lnc', 'h0', 'h1')), ('h0', 'bs', 'lnc', 'h1'), {'lnc': n_prgs, 'h0': H0}
    )
    for lnc_idx in nl.static_range(n_prgs):
        nisa.dma_copy(
            src=src_reshaped[:, :, lnc_idx : lnc_idx + 1, :],
            dst=dst_view.slice(dim=2, start=lnc_idx, end=lnc_idx + 1).get_view(),
        )


def _sb2sb_all_reduce_gather(
    sharded_sb, dtype, replica_group, prg_id: int, n_prgs: int, H0: int, H1: int, H1_shard: int, BxS: int
):
    """SB2SB all-reduce with local gather, returns (output_sb, sharded_AR_sb)."""
    sharded_AR_sb = nl.ndarray(sharded_sb.shape, dtype=dtype, buffer=nl.sbuf)
    nccl.all_reduce(dsts=[sharded_AR_sb], srcs=[sharded_sb], op=nl.add, replica_group=replica_group)

    gathered_sb = nl.ndarray((H0, H1 * BxS), dtype=dtype, buffer=nl.sbuf)
    f_shard = nl.ds(start=prg_id * BxS * H1_shard, size=BxS * H1_shard)
    nisa.tensor_copy(dst=gathered_sb[:, f_shard], src=sharded_AR_sb)

    if n_prgs > 1:
        other_lnc = 1 - prg_id
        f_other_shard = nl.ds(start=other_lnc * BxS * H1_shard, size=BxS * H1_shard)
        nisa.sendrecv(
            src=sharded_AR_sb,
            dst=gathered_sb[:, f_other_shard],
            send_to_rank=other_lnc,
            recv_from_rank=other_lnc,
            pipe_id=0,
        )

    output_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf)
    src_view = TensorView(gathered_sb).rearrange(('h0', ('h1', 'bs')), ('h0', 'bs', 'h1'), {'h1': H1})
    nisa.tensor_copy(dst=output_sb.reshape((H0, BxS, H1)), src=src_view.get_view())

    return output_sb, sharded_AR_sb


# @nki.jit  # Commented out - use nki.jit() at call site to avoid double-jit stack overflow
def transformer_tkg(
    X: nl.ndarray,
    W_qkvs: List[nl.ndarray],
    W_outs: List[nl.ndarray],
    W_gates: List[nl.ndarray],
    W_ups: List[nl.ndarray],
    W_downs: List[nl.ndarray],
    W_gamma_qkvs: List[nl.ndarray],
    W_gamma_mlps: List[nl.ndarray],
    K_caches: List[nl.ndarray],
    V_caches: List[nl.ndarray],
    RoPE_cos: nl.ndarray,
    RoPE_sin: nl.ndarray,
    mask_cache: nl.ndarray,
    mask_active: nl.ndarray,
    position_ids: Optional[nl.ndarray],
    # Config parameters (replacing dataclass)
    num_layers: int,
    eps: float = 1e-6,
    replica_groups: Optional[List[List[int]]] = None,
    sbuf_residual_and_cc: bool = False,
    clamp_bound: float = 0.0,
    # FP8 scales (optional, per layer)
    W_gate_scales: Optional[List[nl.ndarray]] = None,
    W_up_scales: Optional[List[nl.ndarray]] = None,
    W_down_scales: Optional[List[nl.ndarray]] = None,
):
    """
    Transformer token generation forward pass megakernel.

    Performs num_layers transformer layers of the token-generation model.
    Within each layer: attention block, all-reduce CC, MLP, all-reduce CC.
    TODO: Specify intended usage range (e.g., sequence length, batch size)

    Dimensions:
        B: Batch size
        S_tkg: Token generation sequence length (number of new tokens)
        H: Hidden dimension (must be multiple of 128)
        H0: Partition tile size (pmax = 128)
        H1: H // H0
        H1_shard: H1 // n_prgs (per-core shard of hidden dimension)

    Args:
        X (nl.ndarray): [B, S_tkg, H], Input hidden states on HBM
        W_qkvs (List[nl.ndarray]): Per-layer QKV projection weights
        W_outs (List[nl.ndarray]): Per-layer output projection weights
        W_gates (List[nl.ndarray]): Per-layer MLP gate projection weights
        W_ups (List[nl.ndarray]): Per-layer MLP up projection weights
        W_downs (List[nl.ndarray]): Per-layer MLP down projection weights
        W_gamma_qkvs (List[nl.ndarray]): Per-layer RMSNorm gamma for QKV
        W_gamma_mlps (List[nl.ndarray]): Per-layer RMSNorm gamma for MLP
        K_caches (List[nl.ndarray]): Per-layer K caches on HBM
        V_caches (List[nl.ndarray]): Per-layer V caches on HBM
        RoPE_cos (nl.ndarray): [d_head//2, B, S_tkg], RoPE cosine embeddings
        RoPE_sin (nl.ndarray): [d_head//2, B, S_tkg], RoPE sine embeddings
        mask_cache (nl.ndarray): Attention mask for cached KV context
        mask_active (nl.ndarray): Attention mask for active tokens
        position_ids (Optional[nl.ndarray]): [B, 1], KV cache write positions (None = skip cache update)
        num_layers (int): Number of transformer layers to execute
        eps (float): RMSNorm epsilon (default 1e-6)
        replica_groups (Optional[List[List[int]]]): Replica groups for collective communication
        sbuf_residual_and_cc (bool): Use SBUF residual path with SB2SB all-reduce (default False)
        clamp_bound (float): FP8 quantization clipping boundary (default 0.0, 0 = no clipping)
        W_gate_scales (Optional[List[nl.ndarray]]): Per-layer FP8 gate weight scales
        W_up_scales (Optional[List[nl.ndarray]]): Per-layer FP8 up weight scales
        W_down_scales (Optional[List[nl.ndarray]]): Per-layer FP8 down weight scales

    Returns:
        output (nl.ndarray): [B, S_tkg, H], Final hidden states after all transformer layers

    Pseudocode:
        current = X
        for layer_idx in range(num_layers):
            # Step 1: Attention block (RMSNorm + QKV + RoPE + Attention + Output Projection)
            attn_out = attention_block_tkg(current, W_qkv[layer_idx], ...)

            # Step 2: All-reduce across tensor-parallel ranks
            attn_out = all_reduce(attn_out)

            # Step 3: Residual connection
            current = current + attn_out

            # Step 4: MLP block (RMSNorm + Gate/Up projection + SiLU + Down projection)
            mlp_out = mlp(current, W_gate[layer_idx], W_up[layer_idx], W_down[layer_idx], ...)

            # Step 5: All-reduce across tensor-parallel ranks
            mlp_out = all_reduce(mlp_out)

            # Step 6: Residual connection
            current = current + mlp_out
        return current
    """
    B, S_tkg, H = X.shape
    dtype = X.dtype

    # ========== LNC2 Initialization ==========
    _, n_prgs, prg_id = get_verified_program_sharding_info("transformer_tkg", (0, 1), 2)

    # Dimension constants
    H0 = nl.tile_size.pmax
    H1 = H // H0
    H1_shard = H1 // n_prgs
    BxS = B * S_tkg

    # Determine quantization type
    rg = nccl.ReplicaGroup(replica_groups) if replica_groups != None else None

    sbm = BufferManager(0, _SBM_SIZE_BYTES, Logger("transformer_tkg"))
    sbm.set_auto_alloc(False)

    # ==================== Main Loop ====================
    current = X

    for layer_idx in range(num_layers):
        W_qkv = W_qkvs[layer_idx]
        W_out = W_outs[layer_idx]
        W_gate = W_gates[layer_idx]
        W_up = W_ups[layer_idx]
        W_down = W_downs[layer_idx]
        W_gamma_qkv = W_gamma_qkvs[layer_idx]
        W_gamma_mlp = W_gamma_mlps[layer_idx]
        K_cache = K_caches[layer_idx]
        V_cache = V_caches[layer_idx]
        W_gate_scale = W_gate_scales[layer_idx] if W_gate_scales else None
        W_up_scale = W_up_scales[layer_idx] if W_up_scales else None
        W_down_scale = W_down_scales[layer_idx] if W_down_scales else None

        quant_type = QuantizationType.ROW if W_gate_scale else QuantizationType.NONE

        if sbuf_residual_and_cc:
            # ========== SBUF Residual Path ==========
            sbm.set_name_prefix(f"L{layer_idx}_attn_")
            attn_in_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf)
            _load_input_to_sbuf(attn_in_sb, current, BxS, H0, H1, H1_shard, n_prgs)

            residual_attn_in_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=residual_attn_in_sb, src=attn_in_sb)

            X_sb = attn_in_sb.reshape((H0, BxS, H1))
            sbm.set_auto_alloc(True)
            attn_result = attention_block_tkg(
                X=X_sb,
                X_hidden_dim_actual=H,
                rmsnorm_X_enabled=True,
                rmsnorm_X_eps=eps,
                rmsnorm_X_gamma=W_gamma_qkv,
                W_qkv=W_qkv,
                bias_qkv=None,
                quantization_type_qkv=QuantizationType.NONE,
                weight_dequant_scale_qkv=None,
                input_dequant_scale_qkv=None,
                rmsnorm_QK_pre_rope_enabled=False,
                rmsnorm_QK_pre_rope_eps=eps,
                rmsnorm_QK_pre_rope_W_Q=None,
                rmsnorm_QK_pre_rope_W_K=None,
                cos=RoPE_cos,
                sin=RoPE_sin,
                rope_contiguous_layout=True,
                rmsnorm_QK_post_rope_enabled=False,
                rmsnorm_QK_post_rope_eps=eps,
                rmsnorm_QK_post_rope_W_Q=None,
                rmsnorm_QK_post_rope_W_K=None,
                K_cache_transposed=True,
                active_blocks_table=None,
                K_cache=K_cache,
                V_cache=V_cache,
                attention_mask=mask,
                sink=None,
                update_cache=position_ids != None,
                kv_cache_update_idx=position_ids,
                W_out=W_out,
                bias_out=None,
                quantization_type_out=QuantizationType.NONE,
                weight_dequant_scale_out=None,
                input_dequant_scale_out=None,
                transposed_out=True,
                out_in_sb=True,
                sbm=sbm,
            )
            attn_kernel_out_sb = attn_result[0]

            # Free attention block's heap allocations so MLP has full SBM space
            while sbm.heap:
                sbm.pop_heap()

            attn_transformed_sb = attn_kernel_out_sb.reshape((H0, H1_shard * BxS))
            attn_layer_out_sb = _sb2sb_all_reduce_gather(
                attn_transformed_sb, dtype, rg, prg_id, n_prgs, H0, H1, H1_shard, BxS
            )[0]

            mlp_in_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=mlp_in_sb, data1=residual_attn_in_sb, data2=attn_layer_out_sb, op=nl.add)

            residual_mlp_in_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=residual_mlp_in_sb, src=mlp_in_sb)

            sbm.set_name_prefix(f"L{layer_idx}_mlp_")
            mlp_in_reshaped = mlp_in_sb.reshape((H0, BxS, H1))
            sbm.set_auto_alloc(False)
            mlp_outputs = mlp(
                hidden_tensor=mlp_in_reshaped,
                gate_proj_weights_tensor=W_gate,
                up_proj_weights_tensor=W_up,
                down_proj_weights_tensor=W_down,
                normalization_weights_tensor=W_gamma_mlp,
                normalization_type=NormType.RMS_NORM,
                activation_fn=ActFnType.SiLU,
                eps=eps,
                quantization_type=quant_type,
                gate_w_scale=W_gate_scale,
                up_w_scale=W_up_scale,
                down_w_scale=W_down_scale,
                quant_clipping_bound=clamp_bound,
                store_output_in_sbuf=True,
                use_tkg_down_proj_column_tiling=False,
                sbm=sbm,
            )
            mlp_result = mlp_outputs[0]

            mlp_kernel_out_sb = mlp_result.reshape((H0, H1_shard * BxS))
            mlp_layer_out_sb = _sb2sb_all_reduce_gather(
                mlp_kernel_out_sb, dtype, rg, prg_id, n_prgs, H0, H1, H1_shard, BxS
            )[0]

            output_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=output_sb, data1=residual_mlp_in_sb, data2=mlp_layer_out_sb, op=nl.add)

            layer_output = sbm.alloc((B, S_tkg, H), dtype=dtype, buffer=nl.shared_hbm, name="layer_output")
            if prg_id == 0:
                _store_output_to_hbm(layer_output, output_sb, BxS, H0, H1, H1_shard, n_prgs)

            if n_prgs > 1:
                nisa.core_barrier(data=layer_output, cores=(0, 1))

        else:
            # ========== HBM Path ==========
            sbm.set_name_prefix(f"L{layer_idx}_attn_")
            sbm.set_auto_alloc(True)
            attn_result = attention_block_tkg(
                X=current,
                X_hidden_dim_actual=H,
                rmsnorm_X_enabled=True,
                rmsnorm_X_eps=eps,
                rmsnorm_X_gamma=W_gamma_qkv,
                W_qkv=W_qkv,
                bias_qkv=None,
                quantization_type_qkv=QuantizationType.NONE,
                weight_dequant_scale_qkv=None,
                input_dequant_scale_qkv=None,
                rmsnorm_QK_pre_rope_enabled=False,
                rmsnorm_QK_pre_rope_eps=eps,
                rmsnorm_QK_pre_rope_W_Q=None,
                rmsnorm_QK_pre_rope_W_K=None,
                cos=RoPE_cos,
                sin=RoPE_sin,
                rope_contiguous_layout=True,
                rmsnorm_QK_post_rope_enabled=False,
                rmsnorm_QK_post_rope_eps=eps,
                rmsnorm_QK_post_rope_W_Q=None,
                rmsnorm_QK_post_rope_W_K=None,
                K_cache_transposed=True,
                active_blocks_table=None,
                K_cache=K_cache,
                V_cache=V_cache,
                attention_mask=mask_cache,
                sink=None,
                update_cache=position_ids != None,
                kv_cache_update_idx=position_ids,
                W_out=W_out,
                bias_out=None,
                quantization_type_out=QuantizationType.NONE,
                weight_dequant_scale_out=None,
                input_dequant_scale_out=None,
                transposed_out=False,
                out_in_sb=False,
                sbm=sbm,
            )
            attn_out = attn_result[0]

            # Free attention block's heap allocations so MLP has full SBM space
            while sbm.heap:
                sbm.pop_heap()

            if n_prgs > 1:
                nisa.core_barrier(data=attn_out, cores=(0, 1))

            # Get the attention output (either reduced or original)
            if rg != None:
                attn_reduced = sbm.alloc(attn_out.shape, dtype=dtype, buffer=nl.shared_hbm, name="attn_reduced")
                nccl.all_reduce(dsts=[attn_reduced], srcs=[attn_out], op=nl.add, replica_group=rg)
                attn_for_residual = attn_reduced
            else:
                attn_for_residual = attn_out

            # Residual add: current + attn_for_residual
            # Load to SBUF, add, store back
            attn_residual = sbm.alloc(current.shape, dtype=dtype, buffer=nl.shared_hbm, name="attn_residual")
            current_sb = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.sbuf)
            attn_sb = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.sbuf)
            result_sb = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=current_sb, src=current.reshape((BxS, H)))
            nisa.dma_copy(dst=attn_sb, src=attn_for_residual)
            nisa.tensor_tensor(dst=result_sb, data1=current_sb, data2=attn_sb, op=nl.add)
            nisa.dma_copy(dst=attn_residual.reshape((BxS, H)), src=result_sb)

            # Use attn_residual for MLP input
            mlp_input = attn_residual

            sbm.set_name_prefix(f"L{layer_idx}_mlp_")
            sbm.set_auto_alloc(False)
            mlp_outputs = mlp(
                hidden_tensor=mlp_input,
                gate_proj_weights_tensor=W_gate,
                up_proj_weights_tensor=W_up,
                down_proj_weights_tensor=W_down,
                normalization_weights_tensor=W_gamma_mlp,
                normalization_type=NormType.RMS_NORM,
                activation_fn=ActFnType.SiLU,
                eps=eps,
                quantization_type=quant_type,
                gate_w_scale=W_gate_scale,
                up_w_scale=W_up_scale,
                down_w_scale=W_down_scale,
                quant_clipping_bound=clamp_bound,
                use_tkg_down_proj_column_tiling=False,
                sbm=sbm,
            )
            mlp_out = mlp_outputs[0]

            if n_prgs > 1:
                nisa.core_barrier(data=mlp_out, cores=(0, 1))

            # Get the MLP output (either reduced or original)
            if rg != None:
                mlp_reduced = sbm.alloc(mlp_out.shape, dtype=dtype, buffer=nl.shared_hbm, name="mlp_reduced")
                nccl.all_reduce(dsts=[mlp_reduced], srcs=[mlp_out], op=nl.add, replica_group=rg)
                mlp_for_residual = mlp_reduced
            else:
                mlp_for_residual = mlp_out

            # Residual add: attn_residual + mlp_for_residual
            # Load to SBUF, add, store back
            layer_output = sbm.alloc(attn_residual.shape, dtype=dtype, buffer=nl.shared_hbm, name="layer_output")
            attn_res_sb = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.sbuf)
            mlp_sb = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.sbuf)
            mlp_result_sb = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=attn_res_sb, src=attn_residual.reshape((BxS, H)))
            nisa.dma_copy(dst=mlp_sb, src=mlp_for_residual.reshape((BxS, H)))
            nisa.tensor_tensor(dst=mlp_result_sb, data1=attn_res_sb, data2=mlp_sb, op=nl.add)
            nisa.dma_copy(dst=layer_output.reshape((BxS, H)), src=mlp_result_sb)

        current = layer_output

    return current
```
