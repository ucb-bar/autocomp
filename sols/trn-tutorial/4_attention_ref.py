@nki.jit
def test(q, k, v):
  """Beta 2 migrated attention kernel (matches `4_attention_test.py` semantics).

  Inputs are shaped (d_head=128, seqlen). Output is shaped (seqlen, d_head).
  """
  d_head, seqlen = q.shape
  assert q.shape == k.shape == v.shape
  assert d_head == nl.tile_size.pmax  # 128
  assert seqlen % 128 == 0
  assert seqlen >= 512

  PMAX = nl.tile_size.pmax                # 128
  Q_TILE = PMAX                           # 128 queries per tile
  K_TILE_SOFTMAX = nl.tile_size.gemm_moving_fmax  # 512 keys per tile for QK
  K_TILE_OUT = PMAX                       # 128 keys per tile for output matmul

  out = nl.ndarray((seqlen, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

  n_q_tiles = seqlen // Q_TILE
  n_k_tiles_softmax = seqlen // K_TILE_SOFTMAX
  n_k_tiles_out = seqlen // K_TILE_OUT

  # Temporary tiles reused per q-tile.
  q_tile = nl.ndarray((PMAX, Q_TILE), dtype=q.dtype, buffer=nl.sbuf)
  k_tile = nl.ndarray((PMAX, K_TILE_SOFTMAX), dtype=k.dtype, buffer=nl.sbuf)

  qk_buf = nl.ndarray((PMAX, seqlen), dtype=nl.float32, buffer=nl.sbuf)
  row_max = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
  norm_buf = nl.ndarray((PMAX, seqlen), dtype=nl.float32, buffer=nl.sbuf)
  exp_buf = nl.ndarray((PMAX, seqlen), dtype=nl.float32, buffer=nl.sbuf)
  sum_exp = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
  inv_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
  scores = nl.ndarray((PMAX, seqlen), dtype=nl.float32, buffer=nl.sbuf)

  scores_tile = nl.ndarray((PMAX, K_TILE_OUT), dtype=nl.float32, buffer=nl.sbuf)
  v_tile = nl.ndarray((PMAX, K_TILE_OUT), dtype=v.dtype, buffer=nl.sbuf)
  attn_out = nl.ndarray((Q_TILE, d_head), dtype=q.dtype, buffer=nl.sbuf)

  for i_q in nl.affine_range(n_q_tiles):
    q_start = i_q * Q_TILE
    q_end = q_start + Q_TILE

    # Load q tile once.
    nisa.dma_copy(dst=q_tile, src=q[0:PMAX, q_start:q_end])

    # Build qk_buf for this q tile in 512-key chunks.
    for i_k in nl.affine_range(n_k_tiles_softmax):
      k_start = i_k * K_TILE_SOFTMAX
      k_end = k_start + K_TILE_SOFTMAX
      nisa.dma_copy(dst=k_tile, src=k[0:PMAX, k_start:k_end])
      # Declare qk_psum inside the loop so each nc_matmul starts fresh (no cross-iteration accumulation).
      qk_psum = nl.ndarray((Q_TILE, K_TILE_SOFTMAX), dtype=nl.float32, buffer=nl.psum)
      nisa.nc_matmul(dst=qk_psum, stationary=q_tile, moving=k_tile)
      nisa.tensor_copy(dst=qk_buf[0:PMAX, k_start:k_end], src=qk_psum)

    # Softmax across seqlen dimension for each query row.
    nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=qk_buf, axis=1, keepdims=True)
    nisa.tensor_scalar(dst=norm_buf, data=qk_buf, op0=nl.subtract, operand0=row_max)
    nisa.activation(dst=exp_buf, op=nl.exp, data=norm_buf)
    nisa.tensor_reduce(dst=sum_exp, op=nl.add, data=exp_buf, axis=1, keepdims=True)
    nisa.reciprocal(dst=inv_sum, data=sum_exp)
    nisa.tensor_scalar(dst=scores, data=exp_buf, op0=nl.multiply, operand0=inv_sum)

    # Attention output: accumulate in sbuf, not psum
    attn_accum = nl.ndarray((Q_TILE, d_head), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=attn_accum, value=0.0)

    scores_t = nl.ndarray((PMAX, Q_TILE),    dtype=nl.float32, buffer=nl.sbuf)
    v_t      = nl.ndarray((PMAX, K_TILE_OUT), dtype=nl.float32, buffer=nl.sbuf)
    tmp_sbuf = nl.ndarray((Q_TILE, d_head), dtype=nl.float32, buffer=nl.sbuf)

    for i_k in nl.affine_range(n_k_tiles_out):
      k_start = i_k * K_TILE_OUT
      k_end   = k_start + K_TILE_OUT

      nisa.tensor_copy(dst=scores_tile, src=scores[0:PMAX, k_start:k_end])
      nisa.dma_copy(dst=v_tile, src=v[0:PMAX, k_start:k_end])

      # Tensor Engine nc_transpose: SBUF -> PSUM.  Declare psum INSIDE the loop
      # so the compiler allocates a fresh slot each iteration (overwrite, not accumulate).
      # Then tensor_copy brings the transposed tile to SBUF for nc_matmul.
      scores_t_psum = nl.ndarray((PMAX, Q_TILE),    dtype=nl.float32, buffer=nl.psum)
      nisa.nc_transpose(dst=scores_t_psum, data=scores_tile)
      nisa.tensor_copy(dst=scores_t, src=scores_t_psum)

      v_t_psum = nl.ndarray((PMAX, K_TILE_OUT), dtype=nl.float32, buffer=nl.psum)
      nisa.nc_transpose(dst=v_t_psum, data=v_tile)
      nisa.tensor_copy(dst=v_t, src=v_t_psum)

      # Declare tmp_psum inside the loop so each nc_matmul starts fresh.
      tmp_psum = nl.ndarray((Q_TILE, d_head), dtype=nl.float32, buffer=nl.psum)
      nisa.nc_matmul(dst=tmp_psum, stationary=scores_t, moving=v_t)
      nisa.tensor_copy(dst=tmp_sbuf, src=tmp_psum)
      nisa.tensor_tensor(dst=attn_accum, data1=attn_accum, data2=tmp_sbuf, op=nl.add)

    # Store result
    nisa.tensor_copy(dst=attn_out, src=attn_accum)
    nisa.dma_copy(dst=out[q_start:q_end, 0:d_head], src=attn_out)

  return out

'''DEPRECATED
@nki.jit
def test(q, k, v):
    """Important: we are optimizing for shape d_head = 128, seq_len = 4096."""
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_STATIONARY = nl.tile_size.gemm_stationary_fmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # load inputs into SBUF
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)

    # Tile along seqlen_q #
    # for this example we assume that seqlen_q is divisible by PMAX and 
    # seqlen_kv is divisible by FMAX_MOVING, otherwise need to use mask or "final multiplication"
    qk = nl.ndarray((seqlen_q // PMAX, seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                     dtype=nl.float32, buffer=nl.psum)
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY): # loop on stationary_free
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING): # loop on moving_free
            # Q @ K, contract along d_head #
            qk[i_tile_q, i_tile_kv, :, :] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

    # Softmax #
    # reduce max along seqlen_k
    row_max = nl.ndarray((nl.par_dim(PMAX), seqlen_q // PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):

        row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            row_max_kv[:, i_tile_kv] = nisa.tensor_reduce(op=nl.max, data=qk[i_tile_q, i_tile_kv], axis=1)
 
        row_max[:, i_tile_q, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1)

    # subtract max from row
    norm_row = nl.ndarray((seqlen_q // PMAX, PMAX, seqlen_kv),
                       dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        norm_buf = nl.ndarray(shape=(nl.par_dim(PMAX), seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            norm_buf[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.tensor_scalar(
                data=qk[i_tile_q, i_tile_kv],
                op0=nl.subtract,
                operand0=row_max[:, i_tile_q, :],
                engine=nisa.vector_engine)
        nl.store(norm_row[i_tile_q], norm_buf[:,:])

    # exponentiation
    exp_row = nl.ndarray((seqlen_q // PMAX, PMAX, seqlen_kv), dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        # norm_buf = nl.ndarray(shape=(nl.par_dim(PMAX), seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        exp_buf = nl.ndarray(shape=(nl.par_dim(PMAX), seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        norm_buf = nl.load(norm_row[i_tile_q])
        exp_buf[:,:] = nisa.activation(op=nl.exp, data=norm_buf)
        nl.store(exp_row[i_tile_q], exp_buf[:,:])

    # sum of exp results
    sum_row = nl.ndarray((nl.par_dim(PMAX), seqlen_q // PMAX), dtype=nl.float32, buffer=nl.sbuf)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        exp_buf = nl.load(exp_row[i_tile_q])
        sum_row[:, i_tile_q] = nisa.tensor_reduce(op=nl.add,
                                                         data=exp_buf,
                                                         axis=1)

    # reciprocal of sum_row, tile shape is [PMAX, seqlen_q // PMAX]
    inverse_sum_row = nisa.reciprocal(data=sum_row)
    
    scores = nl.ndarray((seqlen_q // PMAX, PMAX, seqlen_kv), dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        scores_buf = nl.ndarray(shape=(nl.par_dim(PMAX), seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        exp_buf = nl.load(exp_row[i_tile_q])
        scores_buf[:,:] = nisa.tensor_scalar(data=exp_buf,
                                               op0=nl.multiply,
                                               operand0=inverse_sum_row[:, i_tile_q],
                                               engine=nisa.vector_engine,
                                               dtype=nl.float32)
        nl.store(scores[i_tile_q], scores_buf[:,:])
        
    # v has the wrong layout
    v_t = nl.ndarray((seqlen_kv // PMAX, PMAX, d_head), dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nisa.nc_transpose(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)])          # TensorE
        v_sbuf_t = nl.ndarray((nl.par_dim(PMAX), d_head), dtype=nl.float32, buffer=nl.sbuf)
        v_sbuf_t[:, :] = nisa.tensor_copy(v_psum_t, dtype=nl.float32)                   # ScalarE
        nl.store(v_t[i_tile_kv], v_sbuf_t[:,:])

    # scores has the wrong layout
    # PMAX restriction on both free and partition dimension when performing transpose.
    # scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, seqlen_q // PMAX, PMAX),
    #                            dtype=nl.float32, buffer=nl.sbuf)
    scores_t = nl.ndarray((seqlen_kv // PMAX, seqlen_q // PMAX, PMAX, PMAX), dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_buf = nl.load(scores[i_tile_q, :, nl.ds(i_tile_kv*PMAX, PMAX)])
            scores_psum_t = nisa.nc_transpose(scores_buf) # TensorE
            scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), PMAX), dtype=nl.float32, buffer=nl.sbuf)
            scores_sbuf_t[:, :] = nisa.tensor_copy(scores_psum_t, dtype=nl.float32)    # ScalarE
            nl.store(scores_t[i_tile_kv, i_tile_q, :, :], scores_sbuf_t)

    # scores @ V, contract along seqlen_kv
    # d_head == P_MAX, no need to tile there
    for i_tile_q in nl.affine_range(seqlen_q // PMAX): # loop on stationary free
        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)
        attn_out = nl.ndarray((nl.par_dim(PMAX), d_head),
                           dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
            scores_sbuf_t = nl.load(scores_t[i_tile_kv, i_tile_q, :, :])
            v_sbuf_t = nl.load(v_t[i_tile_kv, :, :])
            attn_out_psum += nisa.nc_matmul(stationary=scores_sbuf_t,
                                            moving=v_sbuf_t)
        attn_out[:, :] = nisa.tensor_copy(attn_out_psum)
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:,:])

    return kernel_out
'''