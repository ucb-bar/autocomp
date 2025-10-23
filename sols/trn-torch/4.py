import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import benchmark_lib

# Setup logger (will be configured by benchmark_lib)
logger = logging.getLogger(__name__)


# ============================================================================
# MODULE DEFINITION - Replace this section to benchmark different modules
# ============================================================================

@torch.no_grad()
class AttentionManual(nn.Module):
    """
    Manual attention implementation.
    
    Computes: softmax(Q^T @ K) @ V^T
    
    Input shapes:
        q: (d_head, seqlen_q)
        k: (d_head, seqlen_kv)
        v: (d_head, seqlen_kv)
    
    Output shape: (seqlen_q, d_head)
    """
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        """
        Args:
            q: Query tensor of shape (d_head, seqlen_q)
            k: Key tensor of shape (d_head, seqlen_kv)
            v: Value tensor of shape (d_head, seqlen_kv)
        
        Returns:
            attn_out: Attention output of shape (seqlen_q, d_head)
        """
        # Q^T @ K -> (seqlen_q, seqlen_kv)
        qk = torch.matmul(q.T, k)
        
        # Softmax with numerical stability
        # Subtract max for numerical stability
        row_max = torch.max(qk, dim=1, keepdim=True)[0]  # (seqlen_q, 1)
        norm_row = qk - row_max  # (seqlen_q, seqlen_kv)
        exp_row = torch.exp(norm_row)  # (seqlen_q, seqlen_kv)
        sum_row = torch.sum(exp_row, dim=1, keepdim=True)  # (seqlen_q, 1)
        scores = exp_row / sum_row  # (seqlen_q, seqlen_kv)
        
        # V transpose
        v_t = v.T  # (seqlen_kv, d_head)
        
        # scores @ V^T -> (seqlen_q, d_head)
        attn_out = torch.matmul(scores, v_t)
        
        return attn_out


@torch.no_grad()
class AttentionBuiltin(nn.Module):
    """
    Attention module using PyTorch's built-in scaled_dot_product_attention.
    
    Computes: softmax(Q^T @ K) @ V^T
    
    Input shapes:
        q: (d_head, seqlen_q)
        k: (d_head, seqlen_kv)
        v: (d_head, seqlen_kv)
    
    Output shape: (seqlen_q, d_head)
    """
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        """
        Args:
            q: Query tensor of shape (d_head, seqlen_q)
            k: Key tensor of shape (d_head, seqlen_kv)
            v: Value tensor of shape (d_head, seqlen_kv)
        
        Returns:
            attn_out: Attention output of shape (seqlen_q, d_head)
        """
        # Reshape inputs from (d_head, seqlen) to (batch, heads, seqlen, d_head)
        # PyTorch's SDPA expects: (batch, num_heads, seqlen, d_head)
        q_reshaped = q.T.unsqueeze(0).unsqueeze(0)  # (1, 1, seqlen_q, d_head)
        k_reshaped = k.T.unsqueeze(0).unsqueeze(0)  # (1, 1, seqlen_kv, d_head)
        v_reshaped = v.T.unsqueeze(0).unsqueeze(0)  # (1, 1, seqlen_kv, d_head)
        
        # Use PyTorch's optimized scaled_dot_product_attention
        # scale=1.0 to match the reference implementation (no scaling by 1/sqrt(d_head))
        attn_out = F.scaled_dot_product_attention(
            q_reshaped, k_reshaped, v_reshaped, 
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0
        )
        
        # Reshape back to (seqlen_q, d_head)
        attn_out = attn_out.squeeze(0).squeeze(0)
        
        return attn_out


# Choose which implementation to use
Attention = AttentionManual  # Change to AttentionManual to test the manual version


def create_model_and_input():
    """
    Create the model and input tensors.
    Replace this function to benchmark different modules.
    
    Configure your model and input parameters here.
    
    Returns:
        model: nn.Module to benchmark
        inputs: Tuple of input tensors for the model. Use a single-element tuple for one input.
    """
    # Configuration - modify these values
    d_head = 128  # Head dimension
    seqlen_q = 2048  # Query sequence length
    seqlen_kv = 2048  # Key/Value sequence length
    
    # Create input tensors with shapes matching numpy reference
    q = torch.randn([d_head, seqlen_q], dtype=torch.float32)
    k = torch.randn([d_head, seqlen_kv], dtype=torch.float32)
    v = torch.randn([d_head, seqlen_kv], dtype=torch.float32)
    
    # Create model
    model = Attention()
    model.eval()
    
    description = f'Attention: d_head={d_head}, seqlen_q={seqlen_q}, seqlen_kv={seqlen_kv}'
    
    # Return inputs as a tuple (three inputs in this case)
    return model, (q, k, v), description

# ============================================================================
# END MODULE DEFINITION
# ============================================================================


if __name__ == "__main__":
    # Create model and inputs
    model, inputs, description = create_model_and_input()
    
    # Run benchmark
    benchmark_lib.run_benchmark(model, inputs, description)
