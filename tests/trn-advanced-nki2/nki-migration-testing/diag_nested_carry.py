"""Test loop-carried state in NESTED affine_range (outer + inner loop)."""
import numpy as np
import torch
from torch_xla.core import xla_model as xm
import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def test_nested(vals):
    """
    outer loop: num_outer iterations
    inner loop: 4 iterations, accumulating running_max with loop-carry
    vals: (128, num_outer*4) - values to max-reduce per (outer, inner)
    Returns (128, num_outer) - one max per outer iteration.
    """
    B_P_SIZE = 128
    num_outer = 4
    num_inner = 4
    out = nl.ndarray((B_P_SIZE, num_outer), dtype=nl.float32, buffer=nl.shared_hbm)
    vals_sbuf = nl.ndarray((B_P_SIZE, num_outer * num_inner), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=vals_sbuf, src=vals[0:B_P_SIZE, 0:num_outer*num_inner])

    for outer_i in nl.affine_range(num_outer):
        running_max = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=running_max, value=-9984.0)

        tile_val = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
        new_max = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)

        for inner_i in nl.affine_range(num_inner):
            col = outer_i * num_inner + inner_i
            nisa.tensor_copy(dst=tile_val, src=vals_sbuf[0:B_P_SIZE, col:col+1])
            nisa.tensor_tensor(dst=new_max, data1=running_max, data2=tile_val, op=nl.maximum)
            nisa.tensor_copy(dst=running_max, src=new_max)

        # Store result
        result_sbuf = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=result_sbuf, src=running_max)
        nisa.dma_copy(dst=out[0:B_P_SIZE, outer_i:outer_i+1], src=result_sbuf)

    return out


if __name__ == "__main__":
    device = xm.xla_device()
    np.random.seed(7)
    vals_np = np.random.randn(128, 16).astype(np.float32) * 10

    vals = torch.from_numpy(vals_np).to(device)
    result = test_nested(vals)
    out = result.cpu().numpy()

    # Expected: for each outer_i, max over vals[:, outer_i*4:(outer_i+1)*4] per row
    expected = np.zeros((128, 4), dtype=np.float32)
    for i in range(4):
        expected[:, i] = vals_np[:, i*4:(i+1)*4].max(axis=1)

    print(f"Nested loop-carry allclose: {np.allclose(out, expected, atol=1e-5)}")
    print(f"Max diff: {np.max(np.abs(out - expected)):.6f}")
    for outer_i in range(4):
        print(f"  outer={outer_i}: row0 expected={expected[0, outer_i]:.4f}, got={out[0, outer_i]:.4f}")
