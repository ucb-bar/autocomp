import numpy as np
import torch
from torch_xla.core import xla_model as xm

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def test(x_in, cos, sin, lnc_shard=False, first_second_half_impl=True):
    d_head, S = x_in.shape
    d_half = d_head // 2

    x_out = nl.ndarray((d_head, S), dtype=x_in.dtype, buffer=nl.shared_hbm)

    x_upper = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    x_lower = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_upper, src=x_in[0:d_half, 0:S])
    nisa.dma_copy(dst=x_lower, src=x_in[d_half:d_head, 0:S])

    sb_cos = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    sb_sin = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sb_cos, src=cos[0:d_half, 0:S])
    nisa.dma_copy(dst=sb_sin, src=sin[0:d_half, 0:S])

    e_cos = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=e_cos, data1=x_upper, data2=sb_cos, op=nl.multiply)

    o_sin = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=o_sin, data1=x_lower, data2=sb_sin, op=nl.multiply)

    o_cos = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=o_cos, data1=x_lower, data2=sb_cos, op=nl.multiply)

    e_sin = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=e_sin, data1=x_upper, data2=sb_sin, op=nl.multiply)

    out_upper = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out_upper, data1=e_cos, data2=o_sin, op=nl.subtract)

    out_lower = nl.ndarray((d_half, S), dtype=x_in.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out_lower, data1=o_cos, data2=e_sin, op=nl.add)

    nisa.dma_copy(dst=x_out[0:d_half, 0:S], src=out_upper)
    nisa.dma_copy(dst=x_out[d_half:d_head, 0:S], src=out_lower)

    return x_out


if __name__ == "__main__":
    x_in_np = np.load("rope_x_in.npy")
    cos_np = np.load("rope_cos.npy")
    sin_np = np.load("rope_sin.npy")

    device = xm.xla_device()
    x_in = torch.from_numpy(x_in_np).to(device=device)
    cos = torch.from_numpy(cos_np).to(device=device)
    sin = torch.from_numpy(sin_np).to(device=device)

    out = test(x_in, cos, sin)

    np.save("out_beta2_rope.npy", out.detach().cpu().numpy())
