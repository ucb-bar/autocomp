@nki.jit
def test(q, k, v,
         kernel_dtype, acc_type,
         num_heads=8,
         seq_len=2048,
         d_head=128
         ):
  """
  The flash attention core function to calcualte self attention for all q tiles with K and V.
  The q, k, and v start in HBM and will be loaded into SBUF as tiles. The block size of K and V
  is defined in the seq_len parameter. Returns o, l, m:
  o: (num_heads, seq_len, d_head) - output attention values for all heads and q tiles
  l: (num_heads, seq_len, 1) - log-sum-exp normalizer for all heads and q tiles
  m: (num_heads, seq_len, 1) - max values for all heads and q tiles
  """
  B_P_SIZE = 128
  B_F_SIZE = 512
  LARGE_TILE_SZ = seq_len
  REDUCTION_TILE = min(2048, LARGE_TILE_SZ // 2)
  num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE
  num_q_tiles = seq_len // B_P_SIZE

  # Create local storage for output, l, and m for all heads and q tiles
  o = nl.ndarray((num_heads, seq_len, d_head), dtype=kernel_dtype, buffer=nl.shared_hbm)
  l = nl.ndarray((num_heads, seq_len, 1), dtype=kernel_dtype, buffer=nl.shared_hbm)
  m = nl.ndarray((num_heads, seq_len, 1), dtype=kernel_dtype, buffer=nl.shared_hbm)

  # Loop over all heads (outer loop)
  for head_idx in nl.affine_range(num_heads):
    # Loop over all q tiles
    for q_tile_idx in nl.affine_range(num_q_tiles):
      # Load q tile from HBM into SBUF
      q_local_tile = nl.load(q[head_idx, q_tile_idx * B_P_SIZE:(q_tile_idx + 1) * B_P_SIZE, :], dtype=kernel_dtype)

      # mask are used to only apply computation to the lower half of the matrix,
      # which reduce the arthimetic intensity by half
      forward_mask = q_tile_idx * B_P_SIZE >= 0

      qk_res_buf = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), buffer=nl.sbuf, dtype=acc_type)
      max_local = nl.ndarray((nl.par_dim(B_P_SIZE), num_k_tile_per_large_tile), dtype=acc_type)
      for k_i in nl.affine_range(num_k_tile_per_large_tile):
        # Load k tile from HBM into SBUF
        k_tile = nl.load(k[head_idx, :, k_i * B_F_SIZE:(k_i + 1) * B_F_SIZE], dtype=kernel_dtype)
        
        qk_psum = nl.zeros((nl.par_dim(B_P_SIZE), B_F_SIZE),
                            dtype=np.float32, buffer=nl.psum)  # (128, 512)
        multiplication_required_selection = k_i * B_F_SIZE <= q_tile_idx * B_P_SIZE
        qk_psum[:, :] += nl.matmul(q_local_tile, k_tile, transpose_x=True,
                                           mask=multiplication_required_selection) # (p(128), 512)

        left_diagonal_selection = q_tile_idx * B_P_SIZE >= (k_i + 1) * B_F_SIZE
        diagonal_and_right_selection = (q_tile_idx * B_P_SIZE < (k_i + 1) * B_F_SIZE) & forward_mask

        q_pos = q_tile_idx * B_P_SIZE + nl.arange(B_P_SIZE)[:, None]
        k_pos = k_i * B_F_SIZE + nl.arange(B_F_SIZE)[None, :]
        pred = q_pos >= k_pos
        # For tiles on and to the right of the diagonal, need to do affine_select.
        # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
        qk_res_buf[:, k_i * B_F_SIZE:(k_i + 1) * B_F_SIZE] = nisa.affine_select(
          pred=pred,
          on_true_tile=qk_psum[:, :], on_false_value=-9984.0, dtype=kernel_dtype,
          mask=diagonal_and_right_selection)

        # For tiles on the left of the diagonal, direct copy, no select required.
        qk_res_buf[:, k_i * B_F_SIZE:(k_i + 1) * B_F_SIZE] = \
          nl.copy(qk_psum[:, :], dtype=kernel_dtype, mask=left_diagonal_selection)

        # Calculate max of the current tile
        max_local[:, k_i] = nisa.tensor_reduce(np.max, qk_res_buf[:, k_i * B_F_SIZE:(k_i + 1) * B_F_SIZE], axis=(1,),
                                            dtype=acc_type, negate=False, mask=forward_mask)

      max_ = nisa.tensor_reduce(np.max, max_local[:, :], axis=(1, ),
                        dtype=acc_type, negate=False, mask=forward_mask)
      nl.store(m[head_idx, q_tile_idx * B_P_SIZE:q_tile_idx * B_P_SIZE+B_P_SIZE, :], value=nl.copy(max_, dtype=kernel_dtype))
      m_current = max_

      p_local = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
      p_partial_sum = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ // REDUCTION_TILE), dtype=acc_type)
      for k_r_i in nl.affine_range(LARGE_TILE_SZ // REDUCTION_TILE):
        # compute exp(qk-max)
        p_local[:, k_r_i * REDUCTION_TILE:(k_r_i + 1) * REDUCTION_TILE] = \
          nisa.activation(np.exp,
                          qk_res_buf[:, k_r_i * REDUCTION_TILE:(k_r_i + 1) * REDUCTION_TILE],
                          bias=-1 * m_current,
                          scale=1.0,
                          dtype=kernel_dtype,
                          mask=forward_mask)

        # Compute partial row-tile sum of exp(qk-max))
        p_partial_sum[:, k_r_i] = nl.sum(p_local[:, k_r_i * REDUCTION_TILE:(k_r_i + 1) * REDUCTION_TILE], axis=1, dtype=acc_type, mask=forward_mask)

      p_local_transposed = nl.ndarray((nl.par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
      for i_p_t in nl.affine_range(LARGE_TILE_SZ // 512):
        p_local_t_tmp = nl.ndarray((nl.par_dim(B_P_SIZE), 512), buffer=nl.psum, dtype=np.float32)
        for i_p_t_local in nl.affine_range(512//128):
          p_local_t_tmp[:, i_p_t_local*128:(i_p_t_local+1)*128] = nisa.nc_transpose(p_local[:, i_p_t*512+i_p_t_local * B_P_SIZE:i_p_t*512+(i_p_t_local+1) * B_P_SIZE], mask=forward_mask)
        p_local_transposed[:, i_p_t * 512:(i_p_t + 1) * 512] = nl.copy(p_local_t_tmp[:, :], dtype=kernel_dtype, mask=forward_mask)

      ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type, mask=forward_mask)
      pv_psum = nl.zeros((nl.par_dim(B_P_SIZE), d_head), dtype=np.float32, buffer=nl.psum)
      for k_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
        # Load v tile from HBM into SBUF
        v_tile = nl.load(v[head_idx, k_i * B_P_SIZE:(k_i + 1) * B_P_SIZE, :], dtype=kernel_dtype)
        
        pv_psum[:, :] += nl.matmul(p_local_transposed[:, k_i * B_P_SIZE:(k_i + 1) * B_P_SIZE],
                                           v_tile,
                                           transpose_x=True,
                                           mask=forward_mask) # (128, 128) (p(Br), d)

      nl.store(o[head_idx, q_tile_idx * B_P_SIZE:q_tile_idx * B_P_SIZE+B_P_SIZE, :], value=nl.copy(pv_psum[:, :], dtype=kernel_dtype))
      nl.store(l[head_idx, q_tile_idx * B_P_SIZE:q_tile_idx * B_P_SIZE+B_P_SIZE, :], value=nl.add(nl.log(ps), max_))
  
  return o, l, m
