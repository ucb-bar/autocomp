"""Compare exp sum computation between beta1 pattern and beta2 pattern."""
import numpy as np
import torch
from torch_xla.core import xla_model as xm
import nki
import nki.isa as nisa
import nki.language as nl

NEG_INF = -9984.0

@nki.jit
def test_exp_sum(qk_vals, m_val):
    """
    qk_vals: (128, 512) bfloat16 — pre-masked qk values (NEG_INF for masked)
    m_val: (128, 1) float32 — per-row max
    Returns: (128, 1) float32 — sum of exp(qk - m)
    """
    B_P_SIZE = 128
    B_F_SIZE = 512

    out = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    qk_sbuf = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=qk_sbuf, src=qk_vals[0:B_P_SIZE, 0:B_F_SIZE])

    m_sbuf = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=m_sbuf, src=m_val[0:B_P_SIZE, 0:1])

    # Compute neg_max = -m
    neg_max = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=neg_max, data=m_sbuf, op0=nl.multiply, operand0=-1.0)

    # exp(qk - m) using activation with bias=neg_max
    exp_vals = nl.ndarray((B_P_SIZE, B_F_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=exp_vals, op=nl.exp, data=qk_sbuf, bias=neg_max, scale=1.0)

    # Sum
    exp_sum = nl.ndarray((B_P_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=exp_sum, op=nl.add, data=exp_vals, axis=1, keepdims=True)

    nisa.dma_copy(dst=out[0:B_P_SIZE, 0:1], src=exp_sum)
    return out


if __name__ == "__main__":
    q_np = np.load("csa2048_q.npy").astype(np.float32)
    k_np = np.load("csa2048_k.npy").astype(np.float32)
    seq_len = 2048

    # Compute QK for q_tile 0
    q_tile = q_np[0:128, :]  # (128, 128)
    k_tile = k_np[:, 0:512]  # (128, 512)
    qk = q_tile @ k_tile  # (128, 512)

    # Apply causal mask: for row i (absolute row), valid k positions are j <= i
    causal_mask = np.zeros((128, 512), dtype=np.float32)
    for i in range(128):
        causal_mask[i, :i+1] = 1.0

    # Apply masking formula (float32)
    qk_masked_f32 = (qk + 9984.0) * causal_mask - 9984.0

    # Quantize to bfloat16
    qk_masked_bf16 = qk_masked_f32.astype(np.float32)
    # Simulate bfloat16: round to bfloat16
    qk_masked_f32_view = qk_masked_f32.view(np.uint32)
    qk_masked_bf16_view = (qk_masked_f32_view >> 16).astype(np.uint16)
    qk_masked_f32_quantized = (qk_masked_bf16_view.astype(np.uint32) << 16).view(np.float32)

    # Per-row max (from bfloat16 values)
    m_ref = qk_masked_f32_quantized.max(axis=1, keepdims=True)

    print("Row 8: qk_masked (quantized):", qk_masked_f32_quantized[8, :10])
    print("Row 8: m_ref=", m_ref[8, 0])
    sum_ref = np.sum(np.exp(qk_masked_f32_quantized[8, :] - m_ref[8, 0]))
    print(f"Row 8: numpy sum_exp={sum_ref:.4f}, l={np.log(sum_ref)+m_ref[8,0]:.4f}")

    # Run NKI kernel
    device = xm.xla_device()
    import struct
    # Convert to bfloat16 tensor for NKI
    qk_bf16_torch = torch.from_numpy(qk_masked_f32_quantized).bfloat16().to(device)
    m_f32_torch = torch.from_numpy(m_ref.astype(np.float32)).to(device)

    result = test_exp_sum(qk_bf16_torch, m_f32_torch)
    exp_sum_nki = result.cpu().numpy()
    l_nki = np.log(exp_sum_nki[:, 0]) + m_ref[:, 0]
    print(f"Row 8: NKI exp_sum={exp_sum_nki[8, 0]:.4f}, l_nki={l_nki[8]:.4f}")
    print(f"Row 14: NKI exp_sum={exp_sum_nki[14, 0]:.4f}, l_nki={l_nki[14]:.4f}")
