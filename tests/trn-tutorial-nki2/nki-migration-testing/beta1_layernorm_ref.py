import math

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


@nki.jit
def ref(input_tensor, epsilon, gamma_vector, beta_vector):
  """Computes LayerNorm (Beta 1 reference).

  Uses nki.isa bn_stats/bn_aggr to compute mean/variance and performs shift/scale.
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
  for i in nl.affine_range(math.ceil(input_tensor.shape[0] / nl.tile_size.pmax)):
    # Load input tile
    input_sb = nl.load(input_tensor[i * nl.tile_size.pmax + i_p_io, i_f_io],
                       mask=(i * nl.tile_size.pmax + i_p_io < num_rows))

    # Tile free dimension of the input tensor by nl.tile_size.bn_stats_fmax,
    # as bn_stats has a free dimension size limit
    i_f_bn = nl.arange(nl.tile_size.bn_stats_fmax)[None, :]
    i_f_stats = nl.arange(6)[None, :]
    num_bn_stats = math.ceil(input_tensor.shape[1] / nl.tile_size.bn_stats_fmax)
    stats_results = nl.ndarray((nl.tile_size.pmax, 6 * num_bn_stats), dtype=np.float32)
    for j in nl.affine_range(num_bn_stats):
      stats_results[i_p_io, j * 6 + i_f_stats] = nisa.bn_stats(
        input_sb[i_p_io, j * nl.tile_size.bn_stats_fmax + i_f_bn],
        mask=(j * nl.tile_size.bn_stats_fmax + i_f_bn < input_tensor.shape[1]),
        dtype=np.float32)

    # Aggregate bn_stats results to compute mean and var
    i_f_aggr = nl.arange(6 * num_bn_stats)[None, :]
    mean_var = nisa.bn_aggr(stats_results[i_p_io, i_f_aggr])
    mean = mean_var[i_p_io, 0]
    var = mean_var[i_p_io, 1]

    # Get reciprocal of sqrt(var + epsilon)
    scale_var = nl.rsqrt(var + epsilon)

    # Combine shift and scale to encourage a fused instruction sequence
    shift_scale_tensor = nisa.tensor_scalar(data=input_sb, op0=np.subtract,
                                            operand0=mean,
                                            op1=np.multiply,
                                            operand1=scale_var)

    # Scale the normalized tile using gamma and add beta
    output_sb = shift_scale_tensor * gamma_sb_bcast + beta_sb_bcast

    nl.store(output_tensor[i * nl.tile_size.pmax + i_p_io, i_f_io], value=output_sb,
             mask=(i * nl.tile_size.pmax + i_p_io < num_rows))

  return output_tensor


if __name__ == "__main__":
  x = np.load("x.npy")
  gamma = np.load("gamma.npy")
  beta = np.load("beta.npy")
  epsilon = np.float32(1.0e-5)

  out = ref(x, epsilon, gamma, beta)
  np.save("out_beta1_layernorm.npy", out)

