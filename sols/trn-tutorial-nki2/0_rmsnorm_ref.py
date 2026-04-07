@nki.jit
def solution(a_tensor, g_tensor):

  # Row tile size (partition limit); column chunk size for nc_matmul (max 128x512)
  TILE_ROWS = 128
  G_BROADCAST_CHUNK_COLS = 512

  # Compute out_tensor = a_tensor / RMS(a_tensor) * g_tensor
  # where RMS(x) = sqrt((1/N) * sum(x^2)) and N = a_tensor.shape[1].
  # Reduction (mean) is along the last (free) dimension.
  out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                          buffer=nl.shared_hbm)

  assert a_tensor.shape[1] == g_tensor.shape[0]

  num_rows = a_tensor.shape[0]
  n_f = a_tensor.shape[1]

  # Load RMSNorm weight once; reused for all row tiles.
  g_tile = nl.ndarray((1, g_tensor.shape[0]), dtype=g_tensor.dtype, buffer=nl.sbuf)
  g_2d = g_tensor.reshape((1, g_tensor.shape[0]))
  nisa.dma_copy(g_tile, src=g_2d)

  # Process 128 rows at a time (tile size limit); tiles are independent.
  for i in nl.affine_range(math.ceil(num_rows / TILE_ROWS)):
    p_start = i * TILE_ROWS
    p_end = min(num_rows, p_start + TILE_ROWS)
    tile_rows = p_end - p_start

    # Load input tile from HBM to on-chip.
    a_tile = nl.ndarray((tile_rows, n_f), dtype=a_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(a_tile, src=a_tensor[p_start:p_end, 0:n_f])

    # Square elements (op must be nki activation/binary, not np.ufunc).
    in_square = nl.ndarray((tile_rows, n_f), dtype=a_tensor.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=in_square, data1=a_tile, data2=a_tile, op=nl.multiply)

    # Sum of squares along last dimension, then scale by 1/N.
    square_sum = nl.ndarray((tile_rows, 1), dtype=a_tensor.dtype, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=square_sum, op=nl.add, data=in_square, axis=1, keepdims=True)
    mean = nl.ndarray((tile_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mean, data=square_sum, op0=nl.multiply, operand0=1.0 / n_f)

    # RMS = sqrt(mean); then 1/RMS for scaling.
    sqrt_mean = nl.ndarray((tile_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sqrt_mean, op=nl.sqrt, data=mean)
    rms_reciprocal = nl.ndarray((tile_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rms_reciprocal, data=sqrt_mean)

    # Scale input: a_tile * (1/RMS); (tile_rows,1) broadcasts in tensor_scalar.
    out_tile = nl.ndarray((tile_rows, n_f), dtype=a_tensor.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=out_tile, data=a_tile, op0=nl.multiply, operand0=rms_reciprocal)

    # Broadcast g_tile (1, n_f) to (tile_rows, n_f) via nc_matmul in column chunks
    # (moving operand must be at most 128x512 per constraint).
    ones = nl.ndarray((1, tile_rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    for j in nl.affine_range((n_f + G_BROADCAST_CHUNK_COLS - 1) // G_BROADCAST_CHUNK_COLS):
      j_start = j * G_BROADCAST_CHUNK_COLS
      j_end = min(j_start + G_BROADCAST_CHUNK_COLS, n_f)
      chunk = j_end - j_start
      g_chunk = nl.ndarray((1, chunk), dtype=g_tensor.dtype, buffer=nl.sbuf)
      nisa.dma_copy(dst=g_chunk, src=g_tile[0:1, j_start:j_end])
      g_bcast_chunk_psum = nl.ndarray((tile_rows, chunk), dtype=g_tensor.dtype, buffer=nl.psum)
      nisa.nc_matmul(dst=g_bcast_chunk_psum, stationary=ones, moving=g_chunk, is_stationary_onezero=True)
      nisa.tensor_tensor(dst=out_tile[0:tile_rows, j_start:j_end],
                         data1=out_tile[0:tile_rows, j_start:j_end],
                         data2=g_bcast_chunk_psum, op=nl.multiply)

    # Write result tile back to HBM.
    nisa.dma_copy(dst=out_tensor[p_start:p_end, 0:n_f], src=out_tile)

  return out_tensor
