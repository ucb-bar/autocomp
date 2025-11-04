@nki.jit
def test(a_input, b_input):
  """NKI kernel to compute element-wise addition of two input tensors

  Args:
      a_input: a first input tensor
      b_input: a second input tensor

  Returns:
      c_output: an output tensor
  """
  # Create output tensor
  c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)

  # Process in tiles of 128x512 due to hardware limitations
  tile_size_x = 128
  tile_size_y = 512

  for i in range(0, a_input.shape[0], tile_size_x):
    for j in range(0, a_input.shape[1], tile_size_y):
      # Generate tensor indices for this tile
      ix = i + nl.arange(tile_size_x)[:, None]
      iy = j + nl.arange(tile_size_y)[None, :]

      # Load input data from device memory (HBM) to on-chip memory (SBUF)
      a_tile = nl.load(a_input[ix, iy])
      b_tile = nl.load(b_input[ix, iy])

      # compute a + b
      c_tile = a_tile + b_tile

      # store the addition results back to device memory (c_output)
      nl.store(c_output[ix, iy], value=c_tile)

  return c_output