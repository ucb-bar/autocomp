@nki.jit
def test(delta: nt.tensor,
         u:     nt.tensor,
         A:     nt.tensor,
         B:     nt.tensor,
         C:     nt.tensor) -> nt.tensor:
    # shapes:
    #   delta, u: (batch_size, channels, seq_len)
    #   A:        (channels, state_size)
    #   B, C:     (batch_size, state_size, seq_len)
    batch_size, channels, seq_len = delta.shape
    _, state_size = A.shape

    # allocate result in shared HBM
    output = nl.ndarray((batch_size, channels, seq_len),
                        dtype=delta.dtype,
                        buffer=nl.shared_hbm)

    # tile parameters
    assert channels % nl.tile_size.pmax == 0
    channel_psize   = nl.tile_size.pmax
    n_channel_tiles = channels // channel_psize

    # PSUM free-dimension limit; process seq_len in PSUM-friendly chunks
    f_tile = min(seq_len, nl.tile_size.psum_fmax)
    n_full_chunks = seq_len // f_tile
    tail = seq_len - n_full_chunks * f_tile

    # outer batch loop
    for i_batch in nl.affine_range(batch_size):
        # tile over the channel dimension
        for i_cht in nl.affine_range(n_channel_tiles):
            ch_off = i_cht * channel_psize

            # hoist A block for this channel tile into SBUF
            A_block = nl.load(A[ch_off:ch_off+channel_psize,
                                 0:state_size])

            # carry per state for scan across F-chunks: shape [P, state_size]
            carry = nl.zeros((nl.par_dim(channel_psize), state_size),
                             dtype=delta.dtype)

            # process full-size F-chunks
            for i_f in nl.sequential_range(n_full_chunks):
                f0 = i_f * f_tile
                f1 = f0 + f_tile

                # Load only the slice needed for this chunk
                delta_chunk = nl.load(delta[i_batch,
                                            ch_off:ch_off+channel_psize,
                                            f0:f1])  # [P, f_tile]
                u_chunk     = nl.load(u[i_batch,
                                        ch_off:ch_off+channel_psize,
                                        f0:f1])      # [P, f_tile]

                # precompute delta * u for this slice
                deltaU_chunk = nisa.tensor_tensor(delta_chunk, u_chunk, op=nl.multiply)

                # accumulator in PSUM for this F-chunk
                acc_psum = nl.zeros((nl.par_dim(channel_psize), f_tile),
                                    dtype=delta.dtype,
                                    buffer=nl.psum)

                # Double-buffered SBUF tiles for B and C rows of this chunk
                B_buf0 = nl.ndarray(shape=(1, f_tile), dtype=B.dtype, buffer=nl.sbuf)
                B_buf1 = nl.ndarray(shape=(1, f_tile), dtype=B.dtype, buffer=nl.sbuf)
                C_buf0 = nl.ndarray(shape=(1, f_tile), dtype=C.dtype, buffer=nl.sbuf)
                C_buf1 = nl.ndarray(shape=(1, f_tile), dtype=C.dtype, buffer=nl.sbuf)

                # Prime first state\'s B/C into ping buffers
                nisa.dma_copy(dst=B_buf0, src=B[i_batch, 0:1, f0:f1])
                nisa.dma_copy(dst=C_buf0, src=C[i_batch, 0:1, f0:f1])

                # accumulate over all states in PSUM
                for i_state in nl.sequential_range(state_size):
                    # A_i as a [P, 1] vector tile
                    A_i = A_block[:, i_state:i_state+1]

                    # step 1&2: exp(delta * A_i) on the current F-chunk
                    deltaA = nisa.activation(op=nl.exp,
                                             data=delta_chunk,
                                             scale=A_i)

                    if (i_state % 2) == 0:
                        # Prefetch next state\'s B/C into the alternate buffers, if any
                        if i_state + 1 < state_size:
                            nisa.dma_copy(dst=B_buf1, src=B[i_batch, (i_state+1):(i_state+2), f0:f1])
                            nisa.dma_copy(dst=C_buf1, src=C[i_batch, (i_state+1):(i_state+2), f0:f1])

                        # step3: (delta * u) * B on the chunk with implicit P-broadcast
                        deltaBu = nl.multiply(deltaU_chunk, B_buf0)

                        # step4: scan over the chunk with per-state initial carry
                        init_vec = carry[:, i_state:i_state+1]  # [P, 1]
                        scan_res = nisa.tensor_tensor_scan(deltaA,
                                                           deltaBu,
                                                           initial=init_vec,
                                                           op0=np.multiply,
                                                           op1=np.add)

                        # update carry for this state (last element of scan result)
                        last_col = scan_res[:, f_tile-1:f_tile]  # [P, 1]
                        carry[:, i_state:i_state+1] = last_col

                        # step5: multiply by C with implicit P-broadcast and accumulate into PSUM
                        scanC = nl.multiply(scan_res, C_buf0)
                        acc_psum += scanC
                    else:
                        # Prefetch next state\'s B/C into the alternate buffers, if any
                        if i_state + 1 < state_size:
                            nisa.dma_copy(dst=B_buf0, src=B[i_batch, (i_state+1):(i_state+2), f0:f1])
                            nisa.dma_copy(dst=C_buf0, src=C[i_batch, (i_state+1):(i_state+2), f0:f1])

                        # step3: (delta * u) * B on the chunk with implicit P-broadcast
                        deltaBu = nl.multiply(deltaU_chunk, B_buf1)

                        # step4: scan over the chunk with per-state initial carry
                        init_vec = carry[:, i_state:i_state+1]  # [P, 1]
                        scan_res = nisa.tensor_tensor_scan(deltaA,
                                                           deltaBu,
                                                           initial=init_vec,
                                                           op0=np.multiply,
                                                           op1=np.add)

                        # update carry for this state (last element of scan result)
                        last_col = scan_res[:, f_tile-1:f_tile]  # [P, 1]
                        carry[:, i_state:i_state+1] = last_col

                        # step5: multiply by C with implicit P-broadcast and accumulate into PSUM
                        scanC = nl.multiply(scan_res, C_buf1)
                        acc_psum += scanC

                # copy the final PSUM accumulator back into SBUF for store
                acc_sbuf = nisa.tensor_copy(acc_psum, engine=nisa.vector_engine)

                # store current F-chunk to HBM
                nl.store(output[i_batch,
                                ch_off:ch_off+channel_psize,
                                f0:f1],
                         acc_sbuf)

            # process tail chunk if exists (size <= psum_fmax)
            if tail > 0:
                f0 = n_full_chunks * f_tile
                f1 = seq_len
                tail_len = tail

                # Load only the tail slice
                delta_chunk = nl.load(delta[i_batch,
                                            ch_off:ch_off+channel_psize,
                                            f0:f1])  # [P, tail_len]
                u_chunk     = nl.load(u[i_batch,
                                        ch_off:ch_off+channel_psize,
                                        f0:f1])      # [P, tail_len]

                # precompute delta * u for this tail slice
                deltaU_chunk = nisa.tensor_tensor(delta_chunk, u_chunk, op=nl.multiply)

                acc_psum_t = nl.zeros((nl.par_dim(channel_psize), tail_len),
                                      dtype=delta.dtype,
                                      buffer=nl.psum)

                # Double-buffered SBUF tiles for tail B and C rows
                B_t0 = nl.ndarray(shape=(1, tail_len), dtype=B.dtype, buffer=nl.sbuf)
                B_t1 = nl.ndarray(shape=(1, tail_len), dtype=B.dtype, buffer=nl.sbuf)
                C_t0 = nl.ndarray(shape=(1, tail_len), dtype=C.dtype, buffer=nl.sbuf)
                C_t1 = nl.ndarray(shape=(1, tail_len), dtype=C.dtype, buffer=nl.sbuf)

                # Prime first state\'s B/C for tail
                nisa.dma_copy(dst=B_t0, src=B[i_batch, 0:1, f0:f1])
                nisa.dma_copy(dst=C_t0, src=C[i_batch, 0:1, f0:f1])

                for i_state in nl.sequential_range(state_size):
                    A_i = A_block[:, i_state:i_state+1]

                    deltaA_t = nisa.activation(op=nl.exp,
                                               data=delta_chunk,
                                               scale=A_i)

                    if (i_state % 2) == 0:
                        if i_state + 1 < state_size:
                            nisa.dma_copy(dst=B_t1, src=B[i_batch, (i_state+1):(i_state+2), f0:f1])
                            nisa.dma_copy(dst=C_t1, src=C[i_batch, (i_state+1):(i_state+2), f0:f1])

                        # implicit P-broadcast multiplies
                        deltaBu_t = nl.multiply(deltaU_chunk, B_t0)

                        init_vec_t = carry[:, i_state:i_state+1]
                        scan_res_t = nisa.tensor_tensor_scan(deltaA_t,
                                                             deltaBu_t,
                                                             initial=init_vec_t,
                                                             op0=np.multiply,
                                                             op1=np.add)

                        last_col_t = scan_res_t[:, tail_len-1:tail_len]
                        carry[:, i_state:i_state+1] = last_col_t

                        scanC_t = nl.multiply(scan_res_t, C_t0)
                        acc_psum_t += scanC_t
                    else:
                        if i_state + 1 < state_size:
                            nisa.dma_copy(dst=B_t0, src=B[i_batch, (i_state+1):(i_state+2), f0:f1])
                            nisa.dma_copy(dst=C_t0, src=C[i_batch, (i_state+1):(i_state+2), f0:f1])

                        # implicit P-broadcast multiplies
                        deltaBu_t = nl.multiply(deltaU_chunk, B_t1)

                        init_vec_t = carry[:, i_state:i_state+1]
                        scan_res_t = nisa.tensor_tensor_scan(deltaA_t,
                                                             deltaBu_t,
                                                             initial=init_vec_t,
                                                             op0=np.multiply,
                                                             op1=np.add)

                        last_col_t = scan_res_t[:, tail_len-1:tail_len]
                        carry[:, i_state:i_state+1] = last_col_t

                        scanC_t = nl.multiply(scan_res_t, C_t1)
                        acc_psum_t += scanC_t

                acc_sbuf_t = nisa.tensor_copy(acc_psum_t, engine=nisa.vector_engine)
                nl.store(output[i_batch,
                                ch_off:ch_off+channel_psize,
                                f0:f1],
                         acc_sbuf_t)

    return output
