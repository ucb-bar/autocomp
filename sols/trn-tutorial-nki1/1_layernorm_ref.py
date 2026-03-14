@nki.jit
def test(input_tensor, epsilon, gamma_vector, beta_vector):
  """Computes LayerNorm.
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
