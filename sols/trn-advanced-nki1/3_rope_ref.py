@nki.jit
def test(x_in, cos, sin, lnc_shard=False, first_second_half_impl=True):
    def RoPE_sbuf(x_in_sb, cos_sb, sin_sb, x_out_sb):
        d_head, S = x_in_sb.shape
        assert d_head <= 128
        assert tuple(cos_sb.shape) == (d_head // 2, S)
        assert x_in_sb.shape == x_out_sb.shape
        assert cos_sb.shape == sin_sb.shape

        # Indices for selecting upper, lower, even-index, odd-index partitions.
        i_upper = nl.arange(d_head // 2)[:, None]
        i_lower = i_upper + d_head // 2

        i_dh = nl.arange(d_head)[:, None]
        i_S = nl.arange(S)[None, :]

        sb_e = x_in_sb[i_upper, i_S]
        sb_o = x_in_sb[i_lower, i_S]

        '''
        for i in range(d_head/2):
            res[2*i]   = embedding[2*i]   * cos[i] - embedding[2*i+1] * sin[i]
            res[2*i+1] = embedding[2*i+1] * cos[i] + embedding[2*i]   * sin[i]
        '''
        e_cos_sin = nl.ndarray((d_head, S), dtype=x_in_sb.dtype, buffer=nl.sbuf)
        e_cos = e_cos_sin[i_upper, i_S]
        e_sin = e_cos_sin[i_lower, i_S]

        o_cos_sin = nl.ndarray((d_head, S), dtype=x_in_sb.dtype, buffer=nl.sbuf)
        o_cos = o_cos_sin[i_upper, i_S]
        o_sin = o_cos_sin[i_lower, i_S]

        e_cos = nisa.tensor_tensor(sb_e, cos_sb, np.multiply)
        o_cos = nisa.tensor_tensor(sb_o, cos_sb, np.multiply)
        e_sin = nisa.tensor_tensor(sb_e, sin_sb, np.multiply)
        o_sin = nisa.tensor_tensor(sb_o, sin_sb, np.multiply)

        x_out_sb[i_upper, i_S] = nisa.tensor_tensor(e_cos, o_sin, np.subtract) # even * cos -  odd * sin
        x_out_sb[i_lower, i_S] = nisa.tensor_tensor(o_cos, e_sin, np.add)      #  odd * cos + even * sin


    d_head, S = x_in.shape
    assert d_head <= 128
    assert tuple(cos.shape) == (d_head // 2, S)
    assert cos.shape == sin.shape

    # Create x_out tensor with same shape and dtype as x_in
    x_out = nl.ndarray((d_head, S), dtype=x_in.dtype, buffer=nl.hbm)

    n_prgs, prg_id = 1, 0

    if lnc_shard and nl.program_ndim() != 0:
        assert nl.program_ndim() == 1, 'RoPE only supports no specialization or specialization along one axis'
        _prgs = nl.num_programs(axes=0)
        prg_id = nl.program_id(axis=0)

    # Indices for selecting upper, lower, even-index, odd-index partitions.
    i_upper = nl.arange(d_head // 2)[:, None]
    i_lower = i_upper + d_head // 2
    i_even =  i_upper if first_second_half_impl else i_upper * 2
    i_odd = i_lower if first_second_half_impl else i_even + 1

    # Tile along the S dimension.
    tile_size_S = S // n_prgs
    assert S % n_prgs == 0, 'The sequence length is not divisible by number of shards.'
    tile_offset_S = tile_size_S * prg_id
    i_dh = nl.arange(d_head)[:, None]
    i_S = nl.arange(tile_size_S)[None, :]

    # Load input tensor.
    x_in_sb = nl.ndarray((d_head, tile_size_S), dtype=x_in.dtype, buffer=nl.sbuf)
    if first_second_half_impl: # We can load the input tensor at once.
        x_in_sb[i_dh, i_S] = nl.load(x_in[i_dh, i_S + tile_offset_S])
    else:
        x_in_sb[i_upper, i_S] = nl.load(x_in[i_even, i_S + tile_offset_S])
        x_in_sb[i_lower, i_S] = nl.load(x_in[i_odd, i_S + tile_offset_S])

    # Allocate separate buffers for cos and sin to avoid alignment issues
    sb_cos = nl.ndarray((d_head // 2, tile_size_S), dtype=x_in.dtype, buffer=nl.sbuf)
    sb_sin = nl.ndarray((d_head // 2, tile_size_S), dtype=x_in.dtype, buffer=nl.sbuf)

    sb_cos[...] = nl.load(cos[i_upper, i_S + tile_offset_S])
    sb_sin[...] = nl.load(sin[i_upper, i_S + tile_offset_S])

    x_out_sb = nl.ndarray((d_head, tile_size_S), dtype=x_in.dtype, buffer=nl.sbuf)

    RoPE_sbuf(x_in_sb, sb_cos, sb_sin, x_out_sb)

    if first_second_half_impl:
        nl.store(x_out[i_dh, i_S + tile_offset_S], x_out_sb)
    else:
        nl.store(x_out[i_even, i_S + tile_offset_S], x_out_sb[i_upper, i_S])
        nl.store(x_out[i_odd, i_S + tile_offset_S], x_out_sb[i_lower, i_S])

    return x_out