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

# Row tile size (partition limit); column chunk size for nc_matmul (max 128x512)
TILE_ROWS = 128
PARAM_BCAST_CHUNK_COLS = 512

# SUBSTITUTE HERE

def nki_layernorm_kernel_v2(input_tensor, epsilon, gamma_vector, beta_vector):
  # Compute LayerNorm:
  #   y = ((x - mean(x)) / sqrt(var(x) + epsilon)) * gamma + beta
  # Reduction (mean/var) is along the last (free) dimension.
  output_tensor = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype,
                             buffer=nl.shared_hbm)

  assert input_tensor.shape[1] == gamma_vector.shape[0] == beta_vector.shape[0]

  num_rows = input_tensor.shape[0]
  n_f = input_tensor.shape[1]

  # Load gamma/beta once; reused for all row tiles.
  gamma_tile = nl.ndarray((1, gamma_vector.shape[0]), dtype=gamma_vector.dtype, buffer=nl.sbuf)
  beta_tile = nl.ndarray((1, beta_vector.shape[0]), dtype=beta_vector.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=gamma_tile, src=gamma_vector.reshape((1, gamma_vector.shape[0])))
  nisa.dma_copy(dst=beta_tile, src=beta_vector.reshape((1, beta_vector.shape[0])))

  # Process 128 rows at a time (tile size limit); tiles are independent.
  for i in nl.affine_range(math.ceil(num_rows / TILE_ROWS)):
    p_start = i * TILE_ROWS
    p_end = min(num_rows, p_start + TILE_ROWS)
    tile_rows = p_end - p_start

    # Load input tile from HBM to on-chip.
    x_tile = nl.ndarray((tile_rows, n_f), dtype=input_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_tile, src=input_tensor[p_start:p_end, 0:n_f])

    # mean(x) and mean(x^2) along last dimension.
    sum_x = nl.ndarray((tile_rows, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=sum_x, op=nl.add, data=x_tile, axis=1, keepdims=True)

    x_square = nl.ndarray((tile_rows, n_f), dtype=input_tensor.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=x_square, data1=x_tile, data2=x_tile, op=nl.multiply)
    sum_x2 = nl.ndarray((tile_rows, 1), dtype=input_tensor.dtype, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=sum_x2, op=nl.add, data=x_square, axis=1, keepdims=True)

    mean = nl.ndarray((tile_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    ex2 = nl.ndarray((tile_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mean, data=sum_x, op0=nl.multiply, operand0=1.0 / n_f)
    nisa.tensor_scalar(dst=ex2, data=sum_x2, op0=nl.multiply, operand0=1.0 / n_f)

    # var(x) = E[x^2] - (E[x])^2
    mean_sq = nl.ndarray((tile_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=mean_sq, data1=mean, data2=mean, op=nl.multiply)
    var = nl.ndarray((tile_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=var, data=ex2, op0=nl.subtract, operand0=mean_sq)

    # inv_std = 1 / sqrt(var + epsilon)
    var_eps = nl.ndarray((tile_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=var_eps, data=var, op0=nl.add, operand0=epsilon)
    sqrt_var = nl.ndarray((tile_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sqrt_var, op=nl.sqrt, data=var_eps)
    inv_std = nl.ndarray((tile_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_std, data=sqrt_var)

    # Normalize: (x - mean) * inv_std
    out_tile = nl.ndarray((tile_rows, n_f), dtype=input_tensor.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=out_tile, data=x_tile, op0=nl.subtract, operand0=mean,
                       op1=nl.multiply, operand1=inv_std)

    # Broadcast gamma/beta to (tile_rows, n_f) in column chunks and apply:
    # out = out * gamma + beta
    ones = nl.ndarray((1, tile_rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    for j in nl.affine_range((n_f + PARAM_BCAST_CHUNK_COLS - 1) // PARAM_BCAST_CHUNK_COLS):
      j_start = j * PARAM_BCAST_CHUNK_COLS
      j_end = min(j_start + PARAM_BCAST_CHUNK_COLS, n_f)
      chunk = j_end - j_start

      gamma_chunk = nl.ndarray((1, chunk), dtype=gamma_vector.dtype, buffer=nl.sbuf)
      beta_chunk = nl.ndarray((1, chunk), dtype=beta_vector.dtype, buffer=nl.sbuf)
      nisa.dma_copy(dst=gamma_chunk, src=gamma_tile[0:1, j_start:j_end])
      nisa.dma_copy(dst=beta_chunk, src=beta_tile[0:1, j_start:j_end])

      gamma_bcast_psum = nl.ndarray((tile_rows, chunk), dtype=gamma_vector.dtype, buffer=nl.psum)
      beta_bcast_psum = nl.ndarray((tile_rows, chunk), dtype=beta_vector.dtype, buffer=nl.psum)
      nisa.nc_matmul(dst=gamma_bcast_psum, stationary=ones, moving=gamma_chunk, is_stationary_onezero=True)
      nisa.nc_matmul(dst=beta_bcast_psum, stationary=ones, moving=beta_chunk, is_stationary_onezero=True)

      nisa.tensor_tensor(dst=out_tile[0:tile_rows, j_start:j_end],
                         data1=out_tile[0:tile_rows, j_start:j_end],
                         data2=gamma_bcast_psum, op=nl.multiply)
      nisa.tensor_tensor(dst=out_tile[0:tile_rows, j_start:j_end],
                         data1=out_tile[0:tile_rows, j_start:j_end],
                         data2=beta_bcast_psum, op=nl.add)

    # Store result tile back to HBM.
    nisa.dma_copy(dst=output_tensor[p_start:p_end, 0:n_f], src=out_tile)

  return output_tensor


''' DEPRECATED
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
    # shift_scale_tensor = (input_sb - mean_var[i_p_stats, i_f_mean]) * scale_var
    shift_scale_tensor = nisa.tensor_scalar(data=input_sb, op0=np.subtract,
                                            operand0=mean,
                                            op1=np.multiply,
                                            operand1=scale_var)
    
    # Scale the normalized tile using gamma and add beta
    output_sb = shift_scale_tensor * gamma_sb_bcast + beta_sb_bcast

    nl.store(output_tensor[i * nl.tile_size.pmax + i_p_io, i_f_io], value=output_sb,
             mask=(i * nl.tile_size.pmax + i_p_io < num_rows))

  return output_tensor
'''

def construct_args(num_rows, num_cols):
  input_tensor = np.random.rand(num_rows, num_cols).astype(np.float32)
  gamma_vector = np.random.rand(num_cols).astype(np.float32)
  beta_vector = np.random.rand(num_cols).astype(np.float32)
  epsilon = 1e-5
  return input_tensor, epsilon, gamma_vector, beta_vector

def test_nki(ref_func, test_func):
  for _ in range(2):
    args = construct_args(4096, 8192)
    result_1 = ref_func(*args)
    result_2 = test_func(*args)
    print("result_1", result_1[:5, :5])
    print("result_2", result_2[:5, :5])
    if not np.allclose(result_1.astype(nl.float32), result_2.astype(nl.float32), atol=1e-3, rtol=5e-3):
      return False
  return True

def benchmark_nki(nki_func):
  # Benchmarking with large matrices to show the differences more clearly
  args = construct_args(4096, 8192)
  bench_func = nki.benchmark(warmup=2, iters=10)(nki_func)
  bench_func(*args)
  latency_res = bench_func.benchmark_result.nc_latency
  p99 = latency_res.get_latency_percentile(99)
  print("Latency: {:.3f} ms (P99)".format(p99 / 1000.0))

if __name__ == "__main__":
  test_result = test_nki(nki_layernorm_kernel_v2, test)
  if not test_result:
    print("Test failed")
    exit(1)
  else:
    benchmark_nki(test)