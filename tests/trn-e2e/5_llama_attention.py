import math

import torch
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np
from torch_neuronx.xla_impl.ops import nki_jit
from neuronxcc.nki.language import nc

import logging
logger = logging.getLogger("Neuron")

# SUBSTITUTE HERE

def autocomp_token_gen_baseline(Q, K, V, past_key_value, attention_mask):
    """
    Assumes active_mask==None, is_prefix_caching==False, is_speculation==False
    """
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def manual_softmax(prior_scores, active_scores):
        """
        simple softmax computation: denominator is the sum of exp over all vocab and only need compute numerator (exp)
        """
        max_score = torch.max(prior_scores, dim=-1, keepdim=True)[0]
        max_active_score = torch.max(active_scores, dim=-1, keepdim=True)[0]
        max_score = torch.maximum(max_score, max_active_score)

        exp_prior = torch.exp(prior_scores - max_score)
        exp_active = torch.exp(active_scores - max_score)
        denominator = exp_prior.sum(dim=-1, keepdim=True) + exp_active.sum(dim=-1, keepdim=True)

        softmax_prior = exp_prior / denominator
        softmax_active = exp_active / denominator
        return softmax_prior, softmax_active

    K_prior = past_key_value[0]
    V_prior = past_key_value[1]
    K_prior = repeat_kv(K_prior, 4)
    V_prior = repeat_kv(V_prior, 4)
    K_prior = K_prior.transpose(2, 3)
    prior_scores = torch.matmul(Q, K_prior) / math.sqrt(64)

    prior_scores = torch.where(
        attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min
    )
    prior_scores = prior_scores.to(torch.float32)
    # ii. active (current/new) KV
    K_active = repeat_kv(K, 4)
    V_active = repeat_kv(V, 4)
    active_scores = torch.matmul(Q, K_active.transpose(2, 3)) / math.sqrt(64)
    active_scores = active_scores.to(torch.float32)

    # iii. attention scores
    softmax_prior, softmax_active = manual_softmax(
        prior_scores, active_scores
    )

    softmax_prior, softmax_active = softmax_prior.to(Q.dtype), softmax_active.to(Q.dtype)
    attn_prior = torch.matmul(softmax_prior, V_prior)
    attn_active = torch.matmul(softmax_active, V_active)
    attn_output = attn_prior + attn_active
    return attn_output

import time

import torch
import torch_xla
import torch_xla.core.xla_model as xm

# Ensure INFO-level perf logs are always visible, even if another module
# configured logging earlier (basicConfig can be a no-op in that case).
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

def get_input_tensors(batch, num_heads, num_kv_heads, head_dim, seqlen_kv, dtype, device):
    Q = torch.randn(batch, num_heads, 1, head_dim, dtype=dtype, device=device)
    K_active = torch.randn(batch, num_kv_heads, 1, head_dim, dtype=dtype, device=device)
    V_active = torch.randn(batch, num_kv_heads, 1, head_dim, dtype=dtype, device=device)
    K_prior = torch.randn(batch, num_kv_heads, seqlen_kv, head_dim, dtype=dtype, device=device)
    V_prior = torch.randn(batch, num_kv_heads, seqlen_kv, head_dim, dtype=dtype, device=device)
    attention_mask = torch.randint(0, 2, (batch, 1, 1, seqlen_kv), dtype=torch.bool, device=device)
    return Q, K_active, V_active, (K_prior, V_prior), attention_mask

def run_compare(dtype=torch.bfloat16):
    torch.manual_seed(0)

    # For Llama-3.2-1B
    batch = 1
    num_heads = 16
    num_kv_heads = 4
    head_dim = 64
    seqlen_kv = 512

    # # For Llama-3.2-3B
    # batch = 4
    # num_heads = 12
    # num_kv_heads = 4
    # head_dim = 128
    # seqlen_kv = 512

    device = xm.xla_device()

    def run_baseline(*args):
        out = autocomp_token_gen_baseline(
            *args
        )
        torch_xla.sync()
        return out

    def run_nki(*args):
        out = test(*args)
        torch_xla.sync()
        return out

    args = get_input_tensors(batch, num_heads, num_kv_heads, head_dim, seqlen_kv, dtype, device)
    baseline_out = run_baseline(*args)
    attn_output = run_nki(*args)

    if not torch.allclose(baseline_out, attn_output, atol=1e-3, rtol=1e-3):
        # Print a small slice to avoid huge dumps
        b0 = baseline_out.cpu()
        a0 = attn_output.cpu()
        print(f"baseline_out[0, 0, 0, :8]: {b0[0, 0, 0, :8]}")
        print(f"attn_output[0, 0, 0, :8]: {a0[0, 0, 0, :8]}")
        diff = (b0 - a0).abs()
        print(f"max_diff: {diff.max()}")
        print(f"mean_diff: {diff.mean()}")
        print("FAIL: autocomp_token_gen_nki does not match baseline")

    # Lightweight perf check (no assertions).
    perf_iters = 50
    if perf_iters > 0:
        args = get_input_tensors(batch, num_heads, num_kv_heads, head_dim, seqlen_kv, dtype, device)
        # Warmup to ensure compilation happens before timing.
        _ = run_baseline(*args)
        _ = run_nki(*args)
        torch_xla.sync()
        t0 = time.time()
        for _ in range(perf_iters):
            baseline_out = run_baseline(*args)
        torch_xla.sync()
        t1 = time.time()
        print(f"baseline_out: {baseline_out.to('cpu')[0, 0, 0, :8]}")
        torch_xla.sync()
        t2 = time.time()
        for _ in range(perf_iters):
            nki_out = run_nki(*args)
        torch_xla.sync()
        t3 = time.time()
        print(f"nki_out: {nki_out.to('cpu')[0, 0, 0, :8]}")
        baseline_ms = (t1 - t0) * 1000.0 / perf_iters
        nki_ms = (t3 - t2) * 1000.0 / perf_iters
        print(f"autocomp perf over {perf_iters} iters — baseline: {baseline_ms:.3f} ms/iter, nki: {nki_ms:.3f} ms/iter")


if __name__ == "__main__":
    run_compare()
