@nki.jit
def test(delta, u, A, B_exp, C_exp):
  """
  B_exp, C_exp are pre-broadcast on the host:
    original B: [batch, state_size, seq_len]
    B_exp:      [batch, state_size * channel_psize, seq_len]
    where rows [s*channel_psize : (s+1)*channel_psize] all equal B[:, s, :]

  This lets us dma_copy a (channel_psize, seq_len_fsize) tile per state,
  satisfying Beta 2's requirement that partition_dim == pmax == 128.
  """
  batch_size, channels, seq_len = delta.shape
  output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype,
                      buffer=nl.shared_hbm)
  _, state_size = A.shape

  channel_psize = nl.tile_size.pmax   # 128
  n_channel_tile = channels // channel_psize
  seq_len_fsize = 512
  n_seq_len_tile = seq_len // seq_len_fsize

  assert channels % channel_psize == 0
  assert seq_len % seq_len_fsize == 0

  for i_batch in nl.affine_range(batch_size):
    for i_channel_tile in nl.affine_range(n_channel_tile):
      channel_start = i_channel_tile * channel_psize

      scanC_accum = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
      nisa.memset(dst=scanC_accum, value=0.0)

      delta_i = nl.ndarray((channel_psize, seq_len), dtype=delta.dtype, buffer=nl.sbuf)
      u_i     = nl.ndarray((channel_psize, seq_len), dtype=u.dtype,     buffer=nl.sbuf)
      nisa.dma_copy(dst=delta_i, src=delta[i_batch, channel_start:channel_start + channel_psize, 0:seq_len])
      nisa.dma_copy(dst=u_i,     src=u[i_batch,     channel_start:channel_start + channel_psize, 0:seq_len])

      # Scratch buffers
      A_i       = nl.ndarray((channel_psize, 1),             dtype=A.dtype,     buffer=nl.sbuf)
      scan_init = nl.ndarray((channel_psize, 1),             dtype=delta.dtype, buffer=nl.sbuf)
      deltaA    = nl.ndarray((channel_psize, seq_len_fsize), dtype=delta.dtype, buffer=nl.sbuf)
      B_i       = nl.ndarray((channel_psize, seq_len_fsize), dtype=delta.dtype, buffer=nl.sbuf)
      C_i       = nl.ndarray((channel_psize, seq_len_fsize), dtype=delta.dtype, buffer=nl.sbuf)
      deltaU    = nl.ndarray((channel_psize, seq_len_fsize), dtype=delta.dtype, buffer=nl.sbuf)
      deltaBu   = nl.ndarray((channel_psize, seq_len_fsize), dtype=delta.dtype, buffer=nl.sbuf)
      scan_res  = nl.ndarray((channel_psize, seq_len_fsize), dtype=delta.dtype, buffer=nl.sbuf)
      scanC     = nl.ndarray((channel_psize, seq_len_fsize), dtype=delta.dtype, buffer=nl.sbuf)

      # static_range: i_state is compile-time constant → i_state * channel_psize is a literal offset
      for i_state in nl.static_range(state_size):
        b_row_start = i_state * channel_psize
        b_row_end   = b_row_start + channel_psize

        nisa.dma_copy(dst=A_i, src=A[channel_start:channel_start + channel_psize, i_state:i_state + 1])
        nisa.memset(dst=scan_init, value=0.0)

        for i_seq_len_tile in nl.static_range(n_seq_len_tile):
          seq_start = i_seq_len_tile * seq_len_fsize
          seq_end   = seq_start + seq_len_fsize

          # Load B/C as (channel_psize, seq_len_fsize) — all partitions hold same value (pre-broadcast)
          nisa.dma_copy(dst=B_i, src=B_exp[i_batch, b_row_start:b_row_end, seq_start:seq_end])
          nisa.dma_copy(dst=C_i, src=C_exp[i_batch, b_row_start:b_row_end, seq_start:seq_end])

          # Step 1&2: deltaA = exp(delta * A)
          nisa.activation(dst=deltaA, op=nl.exp,
                          data=delta_i[0:channel_psize, seq_start:seq_end],
                          scale=A_i)

          # Step 3: deltaBu = delta * u * B
          nisa.tensor_tensor(dst=deltaU,
                             data1=delta_i[0:channel_psize, seq_start:seq_end],
                             data2=u_i[0:channel_psize, seq_start:seq_end],
                             op=nl.multiply)
          nisa.tensor_tensor(dst=deltaBu, data1=deltaU, data2=B_i, op=nl.multiply)

          # Step 4: Associative scan
          nisa.tensor_tensor_scan(dst=scan_res, data0=deltaA, data1=deltaBu,
                                  initial=scan_init, op0=nl.multiply, op1=nl.add)

          # Update carry
          nisa.tensor_copy(dst=scan_init, src=scan_res[0:channel_psize, seq_len_fsize - 1:seq_len_fsize])

          # Step 5: scanC = scan_res * C
          nisa.tensor_tensor(dst=scanC, data1=scan_res, data2=C_i, op=nl.multiply)

          # Step 6: Accumulate across states
          nisa.tensor_tensor(
              dst=scanC_accum[0:channel_psize, seq_start:seq_end],
              data1=scanC_accum[0:channel_psize, seq_start:seq_end],
              data2=scanC,
              op=nl.add,
              engine=nisa.vector_engine
          )

      nisa.dma_copy(dst=output[i_batch, channel_start:channel_start + channel_psize, 0:seq_len],
                    src=scanC_accum[0:channel_psize, 0:seq_len])

  return output

