@nki.jit
def test(a_tensor):
  # Calculate out_tensor
  # Where softmax(x) = = exp(x - max(x)) / sum(exp(x - max(x)))
  out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                          buffer=nl.shared_hbm)

  # Generate tensor indices to index input tensor
  ix = nl.arange(128)[:, None]
  iy = nl.arange(a_tensor.shape[1])[None, :]

  num_rows = a_tensor.shape[0]

  # Process 128 rows at a time due to 128-partition tile size limitation
  # Since we're not reducing across the first dimension
  # Tiles can be processed independently
  for i in nl.affine_range(math.ceil(a_tensor.shape[0]/128)):

    # Load input data from external memory to on-chip memory
    a_tile = nl.load(a_tensor[i * 128 + ix, iy],
                    mask=(i * 128 + ix < num_rows))

    # Find max and subtract from each value to ensure numerical stability
    max_vals = nl.max(a_tile, axis=[1], keepdims=True, mask=(i * 128 + ix < num_rows))
    shifted = nl.subtract(a_tile, max_vals, mask=(i * 128 + ix < num_rows))

    # Compute element-wise exp of a_tensor
    numerator = nl.exp(shifted)

    # Calculate sum of squared elements, along last dimension
    denominator = nl.sum(numerator, axis=[1])

    # Scale and get a reciprocal
    sm = numerator / denominator

    # store the results back to external memory (out_tensor)
    nl.store(out_tensor[i * 128 + ix, iy], value=sm,
            mask=(i * 128 + ix < num_rows))

  return out_tensor