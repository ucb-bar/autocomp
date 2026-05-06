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