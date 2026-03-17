"""Test tensor_copy reading from sbuf with dynamic free-dim offset in inner loop."""
import numpy as np
import torch
from torch_xla.core import xla_model as xm
import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def test_read_offset(data):
    """
    data: (128, 512) -- a 128x512 tensor.
    For each sub_i in [0,1,2,3], read data[0:128, sub_i*128:(sub_i+1)*128] and sum.
    Return the total sum per row.
    """
    B_P_SIZE = 128
    B_F_SIZE = 512
    out = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    data_sbuf = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=data_sbuf, src=data[0:B_P_SIZE, 0:B_F_SIZE])

    accum = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=accum, value=0.0)

    slice_buf = nl.ndarray((B_P_SIZE, 128), dtype=nl.float32, buffer=nl.sbuf)
    slice_sum = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)

    for sub_i in nl.affine_range(4):
        sub_start = sub_i * 128
        # Read slice with dynamic offset
        nisa.tensor_copy(dst=slice_buf, src=data_sbuf[0:B_P_SIZE, sub_start:sub_start+128])
        # Sum the slice
        nisa.tensor_reduce(dst=slice_sum, op=nl.add, data=slice_buf, axis=1, keepdims=True)
        # Accumulate
        tmp = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=tmp, data1=accum, data2=slice_sum, op=nl.add)
        nisa.tensor_copy(dst=accum, src=tmp)

    nisa.dma_copy(dst=out[0:B_P_SIZE, 0:1], src=accum)
    return out


if __name__ == "__main__":
    device = xm.xla_device()
    np.random.seed(42)
    data_np = np.random.randn(128, 512).astype(np.float32)
    data = torch.from_numpy(data_np).to(device)

    result = test_read_offset(data)
    out = result.cpu().numpy()

    expected = data_np.sum(axis=1, keepdims=True)
    print(f"Read-offset sum allclose: {np.allclose(out, expected, atol=1e-3)}")
    print(f"Max diff: {np.max(np.abs(out - expected)):.6f}")
    print(f"Row 0: expected={expected[0,0]:.4f}, got={out[0,0]:.4f}")
    print(f"Row 1: expected={expected[1,0]:.4f}, got={out[1,0]:.4f}")
