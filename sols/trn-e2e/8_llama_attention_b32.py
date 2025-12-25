@nki.jit
def autocomp_token_gen_nki(Q, K, V, past_key_value, attention_mask):
    """
    Token-generation attention kernel in NKI. Not yet implemented; we want to translate torch code below to NKI.

    Assumes:
    - active_mask is None
    - is_prefix_caching is False
    - is_speculation is False
    Shapes follow the baseline path:
      Q: [batch, num_heads, 1, head_dim]
      K/V: [batch, num_kv_heads, 1, head_dim]
      past_key_value[0/1]: [batch, num_kv_heads, seqlen_kv, head_dim]
      attention_mask: [batch, num_heads, 1, seqlen_kv] (bool)
      attn_output: shape varies depends on what is being computed
    We are using constant shapes.
      batch = 32, num_heads = 16, num_kv_heads = 4, head_dim = 64, seqlen_kv = 512
    """
    kernel_output = nl.ndarray(Q.shape, dtype=Q.dtype, buffer=nl.hbm)
    out = nl.zeros(kernel_output.shape, dtype=kernel_output.dtype)
    nl.store(kernel_output, out)
    return kernel_output

import torch
import math

def test(Q, K, V, past_key_value, attention_mask):
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
    
    # kernel_output = torch.zeros_like(Q)
    kernel_output = autocomp_token_gen_nki(
        Q, K, V, past_key_value, attention_mask
    )

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
