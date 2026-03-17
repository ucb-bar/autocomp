"""Test loop-carried SBUF state in affine_range — running max accumulation."""
import numpy as np
import torch
from torch_xla.core import xla_model as xm
import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def test_loop_carry(vals):
    """
    Accumulate a running max across affine_range iterations.
    vals: (128, 4) — 4 values per row, one per k_i.
    Returns the running max after all 4 iterations.
    """
    B_P_SIZE = 128

    max_out = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    vals_sbuf = nl.ndarray((B_P_SIZE, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=vals_sbuf, src=vals[0:B_P_SIZE, 0:4])

    running_max = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=running_max, value=-9984.0)

    tile_val = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    new_max = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)

    for k_i in nl.affine_range(4):
        # Extract vals_sbuf[:, k_i] into tile_val
        nisa.tensor_copy(dst=tile_val, src=vals_sbuf[0:B_P_SIZE, k_i:k_i+1])
        # Update running_max = max(running_max, tile_val)
        nisa.tensor_tensor(dst=new_max, data1=running_max, data2=tile_val, op=nl.maximum)
        nisa.tensor_copy(dst=running_max, src=new_max)

    nisa.dma_copy(dst=max_out[0:B_P_SIZE, 0:1], src=running_max)
    return max_out


if __name__ == "__main__":
    device = xm.xla_device()
    np.random.seed(42)
    vals_np = np.random.randn(128, 4).astype(np.float32) * 10

    vals = torch.from_numpy(vals_np).to(device)
    result = test_loop_carry(vals)
    out = result.cpu().numpy()

    expected = vals_np.max(axis=1, keepdims=True)
    print(f"Loop-carry running max allclose: {np.allclose(out, expected, atol=1e-5)}")
    print(f"Max diff: {np.max(np.abs(out - expected)):.6f}")
    print(f"Sample (row 0): expected={expected[0,0]:.4f}, got={out[0,0]:.4f}")
    print(f"Sample (row 1): expected={expected[1,0]:.4f}, got={out[1,0]:.4f}")
    # Show first few vals
    for i in range(3):
        print(f"  row {i} vals: {vals_np[i]}, max={expected[i,0]:.4f}, got={out[i,0]:.4f}")
