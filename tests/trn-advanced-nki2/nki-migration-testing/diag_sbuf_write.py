"""Test whether nisa.tensor_copy(dst=sbuf[0:N, k_start:k_end], src=sbuf2) writes to the correct offset."""
import numpy as np
import torch
from torch_xla.core import xla_model as xm
import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def test_write_offsets(inp):
    """
    For each of 4 tiles of width 512, write a distinct fill value into a (128, 2048) buffer.
    Then read back the full buffer and return it.
    inp is (128, 4) - one distinct value per tile per row.
    """
    B_P_SIZE = 128
    B_F_SIZE = 512
    seq_len = 2048
    num_k_tiles = seq_len // B_F_SIZE

    out = nl.ndarray((B_P_SIZE, seq_len), dtype=nl.float32, buffer=nl.shared_hbm)

    # The large accumulation buffer
    buf = nl.ndarray((B_P_SIZE, seq_len), dtype=nl.float32, buffer=nl.sbuf)

    inp_sbuf = nl.ndarray((B_P_SIZE, num_k_tiles), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=inp_sbuf, src=inp[0:B_P_SIZE, 0:num_k_tiles])

    tile_buf = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.sbuf)

    for k_i in nl.affine_range(num_k_tiles):
        k_start = k_i * B_F_SIZE
        k_end = k_start + B_F_SIZE
        # Fill tile_buf with inp_sbuf[:, k_i] value (one scalar per row)
        # Use tensor_scalar: tile_buf = inp_sbuf[:, k_i] * ones
        # Actually, use activation: exp(0 + inp_sbuf[:, k_i]) but that changes value.
        # Simpler: use tensor_reduce to get one scalar then broadcast? No.
        # Just use memset with a fixed value per iteration to test the offset mechanism.
        # We'll use k_i * 100.0 as the fill value.
        val = k_i * 100.0 + 1.0  # 1, 101, 201, 301
        nisa.memset(dst=tile_buf, value=val)
        # Write tile_buf to buf at dynamic offset
        nisa.tensor_copy(dst=buf[0:B_P_SIZE, k_start:k_end], src=tile_buf)

    # Read back the full buffer to out via dma_copy
    nisa.dma_copy(dst=out[0:B_P_SIZE, 0:seq_len], src=buf)
    return out


if __name__ == "__main__":
    device = xm.xla_device()
    inp_np = np.zeros((128, 4), dtype=np.float32)
    inp = torch.from_numpy(inp_np).to(device)

    result = test_write_offsets(inp)
    out = result.cpu().numpy()

    print("Output shape:", out.shape)
    # Expected: col 0:512 = 1.0, col 512:1024 = 101.0, col 1024:1536 = 201.0, col 1536:2048 = 301.0
    for k_i in range(4):
        k_start = k_i * 512
        k_end = k_start + 512
        expected_val = k_i * 100.0 + 1.0
        actual_slice = out[0, k_start:k_end]
        match = np.allclose(actual_slice, expected_val)
        print(f"k_i={k_i} (cols {k_start}:{k_end}): expected={expected_val}, actual_mean={actual_slice.mean():.1f}, match={match}")
