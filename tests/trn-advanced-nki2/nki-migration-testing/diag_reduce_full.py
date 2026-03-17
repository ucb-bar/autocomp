"""Test nisa.tensor_reduce on full (128, 2048) sbuf after writing distinct values per 512-chunk."""
import numpy as np
import torch
from torch_xla.core import xla_model as xm
import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def test_reduce(vals):
    """
    Write val[k_i] to each 512-column block of a (128, 2048) sbuf.
    Then tensor_reduce(max) over the full buffer and return.
    vals is (128, 4) - one value per tile per row.
    """
    B_P_SIZE = 128
    B_F_SIZE = 512
    seq_len = 2048
    num_k_tiles = seq_len // B_F_SIZE

    max_out = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    buf = nl.ndarray((B_P_SIZE, seq_len), dtype=nl.float32, buffer=nl.sbuf)
    tile = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    vals_sbuf = nl.ndarray((B_P_SIZE, num_k_tiles), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=vals_sbuf, src=vals[0:B_P_SIZE, 0:num_k_tiles])

    for k_i in nl.affine_range(num_k_tiles):
        k_start = k_i * B_F_SIZE
        k_end = k_start + B_F_SIZE
        # Fill tile with 10^k_i (k_i=0: 1.0, k_i=1: 10.0, k_i=2: 100.0, k_i=3: 1000.0)
        fill_val = 10.0 ** k_i
        nisa.memset(dst=tile, value=fill_val)
        nisa.tensor_copy(dst=buf[0:B_P_SIZE, k_start:k_end], src=tile)

    # Full tensor reduce
    max_sbuf = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=max_sbuf, op=nl.maximum, data=buf, axis=1, keepdims=True)
    nisa.dma_copy(dst=max_out[0:B_P_SIZE, 0:1], src=max_sbuf)
    return max_out


if __name__ == "__main__":
    device = xm.xla_device()
    vals_np = np.zeros((128, 4), dtype=np.float32)
    vals = torch.from_numpy(vals_np).to(device)

    result = test_reduce(vals)
    out = result.cpu().numpy()

    # Expected: max over [1.0, 10.0, 100.0, 1000.0] = 1000.0 for all rows
    expected = 1000.0
    print(f"Expected max: {expected}")
    print(f"Actual max (row 0): {out[0, 0]}")
    print(f"All rows match {expected}: {np.allclose(out, expected)}")
    print(f"Sample values: {out[:5, 0]}")
