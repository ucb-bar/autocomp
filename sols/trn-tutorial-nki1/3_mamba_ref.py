@nki.jit
def mamba_v3(delta, u, A, B, C):
    """Computes the SSM operation in the Mamba model.

    :param delta: (batch_size, channels, seq_len)
    :param u: (batch_size, channels, seq_len)
    :param A: (channels, state_size)
    :param B: (batch_size, state_size, seq_len)
    :param C: (batch_size, state_size, seq_len)
    :return: (batch_size, channels, seq_len)
    """
    batch_size, channels, seq_len = delta.shape
    output = nl.ndarray((batch_size, channels, seq_len), dtype=delta.dtype,
                        buffer=nl.shared_hbm)
    _, state_size = A.shape

    # Map channels to the partition dimension
    # Tile channels to comply with NKI tile size constraints
    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize

    # Magic number, decided through empiracal profiling data
    seq_len_fsize = 512
    n_seq_len_tile = seq_len // seq_len_fsize

    # Fix this later with mask
    assert channels % channel_psize == 0
    assert seq_len % seq_len_fsize == 0

    # Most outer loop with batch_size, parallel_for
    for i_batch in nl.affine_range(batch_size):

        # Second outer loop: tiling channels
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize

            # partial accumulated scanC result with processed states
            scanC_accum = nl.zeros((nl.par_dim(channel_psize), seq_len), dtype=delta.dtype)

            # Load delta/u once to be reused across states
            delta_i = nl.load(delta[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])
            u_i = nl.load(u[i_batch, channel_start:channel_start+channel_psize, 0:seq_len])

            # Inner loop with state_size, partial parallel
            for i_state in nl.affine_range(state_size):
                # Load the relevant tile from A
                A_i = nl.load(A[channel_start:channel_start+channel_psize, i_state])

                # Last scan result
                scan_init = nl.zeros((channel_psize, 1), dtype=delta_i.dtype)

                # FIXME: sequential_range gives incorrect answer and also much worse perf than static_range
                # for i_seq_len_tile in nl.sequential_range(n_seq_len_tile):
                for i_seq_len_tile in nl.static_range(n_seq_len_tile):
                    seq_len_start = i_seq_len_tile * seq_len_fsize

                    # Step 1&2: Element-wise multiplication of delta_i and A_i and then exponential
                    deltaA = nisa.activation(op=nl.exp,
                            data=delta_i[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize],
                            scale=A_i)

                    # Load the relevant tile from B
                    B_i = nl.load(B[i_batch, i_state:i_state+1, seq_len_start:seq_len_start+seq_len_fsize])

                    # Step 3: Element-wise multiplication of delta_i, B_i and u_i
                    deltaU = nisa.tensor_tensor(delta_i[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize],
                            u_i[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize],
                            op=nl.multiply)
                    B_i_bcast = B_i.broadcast_to((channel_psize, seq_len_fsize))
                    deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=nl.multiply)

                    # Step 4: Associative scan between deltaA and deltaBu
                    scan_res = nki.isa.tensor_tensor_scan(deltaA, deltaBu, initial=scan_init,
                            op0=np.multiply, op1=np.add)
                    scan_init[...] = scan_res[0:channel_psize, seq_len_fsize-1]

                    # Load the relevant tile from C
                    C_i = nl.load(C[i_batch, i_state:i_state+1, seq_len_start:seq_len_start+seq_len_fsize])

                    # Step 5: Element-wise multiplication of scan_res and C_i
                    C_i_bcast = C_i.broadcast_to((channel_psize, seq_len_fsize))
                    scanC = nisa.tensor_tensor(scan_res, C_i_bcast, op=nl.multiply)

                    # Step 6: Accumulation of scanC along state_size dimension
                    scanC_accum[0:channel_psize, seq_len_start:seq_len_start+seq_len_fsize] += scanC

            # Store scanC_accum for a single batch to output
            nl.store(output[i_batch, channel_start:channel_start+channel_psize, 0:seq_len],
                    scanC_accum[0:channel_psize, 0:seq_len])
    return output
