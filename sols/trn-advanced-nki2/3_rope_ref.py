@nki.jit
def solution(x_in, cos, sin, lnc_shard=False, first_second_half_impl=True):
    d_head, S = x_in.shape
    d_half = d_head // 2

    x_out = nl.ndarray((d_head, S), dtype=x_in.dtype, buffer=nl.shared_hbm)

    # Load upper half (rows 0..d_half-1) and lower half (rows d_half..d_head-1)
    x_upper = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    x_lower = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_upper, src=x_in[0:d_half, 0:S])
    nisa.dma_copy(dst=x_lower, src=x_in[d_half:d_head, 0:S])

    sb_cos = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    sb_sin = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sb_cos, src=cos[0:d_half, 0:S])
    nisa.dma_copy(dst=sb_sin, src=sin[0:d_half, 0:S])

    # e_cos = x_upper * cos
    e_cos = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=e_cos, data1=x_upper, data2=sb_cos, op=nl.multiply)

    # o_sin = x_lower * sin
    o_sin = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=o_sin, data1=x_lower, data2=sb_sin, op=nl.multiply)

    # o_cos = x_lower * cos
    o_cos = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=o_cos, data1=x_lower, data2=sb_cos, op=nl.multiply)

    # e_sin = x_upper * sin
    e_sin = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=e_sin, data1=x_upper, data2=sb_sin, op=nl.multiply)

    # upper output = e_cos - o_sin
    out_upper = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out_upper, data1=e_cos, data2=o_sin, op=nl.subtract)

    # lower output = o_cos + e_sin
    out_lower = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out_lower, data1=o_cos, data2=e_sin, op=nl.add)

    nisa.dma_copy(dst=x_out[0:d_half, 0:S], src=out_upper)
    nisa.dma_copy(dst=x_out[d_half:d_head, 0:S], src=out_lower)

    return x_out
