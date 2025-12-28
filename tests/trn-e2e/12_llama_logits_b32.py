import torch
import torch_xla
import torch_xla.core.xla_model as xm
import time

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.typing as nt
import numpy as np

# SUBSTITUTE HERE

def torch_logits_baseline(hidden_states, lm_head_weight):
    """
    Torch baseline for LM head logits computation.
    
    Args:
        hidden_states: (32, 1, 2048)
        lm_head_weight: (2048, 64128)
    Returns:
        logits: (32, 1, 64128)
    """
    # Reshape hidden_states from (32, 1, 2048) to (32, 2048)
    hidden_states_2d = hidden_states.view(32, 2048)
    # Compute: hidden_states @ lm_head_weight -> (32, 2048) @ (2048, 64128) = (32, 64128)
    logits = torch.matmul(hidden_states_2d, lm_head_weight)
    # Reshape back to (32, 1, 64128)
    return logits.view(32, 1, 64128)

def get_input_tensors(batch, seq, hidden_size, vocab_size, dtype, device):
    """Create test input tensors."""
    hidden_states = torch.randn(batch, seq, hidden_size, dtype=dtype, device=device)
    lm_head_weight = torch.randn(hidden_size, vocab_size, dtype=dtype, device=device)
    return hidden_states, lm_head_weight

def run_compare(dtype=torch.bfloat16):
    torch.manual_seed(0)
    
    batch = 32
    seq = 1
    hidden_size = 2048
    vocab_size = 64128
    
    device = xm.xla_device()
    
    def run_baseline(hidden_states, lm_head_weight):
        out = torch_logits_baseline(hidden_states, lm_head_weight)
        torch_xla.sync()
        return out
    
    def run_nki(hidden_states, lm_head_weight):
        out = test(hidden_states, lm_head_weight)
        torch_xla.sync()
        return out
    
    args = get_input_tensors(batch, seq, hidden_size, vocab_size, dtype, device)
    baseline_out = run_baseline(*args)
    nki_out = run_nki(*args)
    
    if not torch.allclose(baseline_out, nki_out, atol=1e-3, rtol=1e-3):
        b0 = baseline_out.cpu()
        n0 = nki_out.cpu()
        diff_l2_norm = torch.linalg.norm(b0 - n0)
        b0_l2_norm = torch.linalg.norm(b0)
        print(f"diff_l2_norm / b0_l2_norm: {diff_l2_norm / b0_l2_norm}")
        if diff_l2_norm / b0_l2_norm < 1e-3:
            print("Failed allclose, but L2 norm of difference is less than 1e-3 of baseline L2 norm")
        else:
            # Print a small slice to avoid huge dumps
            print(f"baseline_out.shape: {b0.shape}")
            print(f"nki_out.shape: {n0.shape}")
            print(f"baseline_out[0, 0, :8]: {b0[0, 0, :8]}")
            print(f"nki_out[0, 0, :8]: {n0[0, 0, :8]}")
            diff = (b0 - n0).abs()
            print(f"max_diff: {diff.max()}")
            print(f"mean_diff: {diff.mean()}")
            print("FAIL: test does not match baseline")
            return False
    
    # Lightweight perf check (no assertions).
    perf_iters = 50
    if perf_iters > 0:
        args = get_input_tensors(batch, seq, hidden_size, vocab_size, dtype, device)
        # Warmup to ensure compilation happens before timing.
        _ = run_nki(*args)
        torch_xla.sync()
        t0 = time.perf_counter()
        for _ in range(perf_iters):
            nki_out = run_nki(*args)
        torch_xla.sync()
        t1 = time.perf_counter()
        nki_ms = (t1 - t0) * 1000.0 / perf_iters
        print("Latency: {:.3f} ms".format(nki_ms))
    
    return True

if __name__ == "__main__":
    success = run_compare()
    if not success:
        exit(1)
    print("Test passed")

