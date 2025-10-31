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
class Attention(nn.Module):
    """
    Manual attention implementation.
    
    Computes: softmax(Q @ K.T) @ V
    
    Input shapes:
        q: (seqlen, d_head)
        k: (seqlen, d_head)
        v: (seqlen, d_head)
    
    Output shape: (seqlen, d_head)
    """
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        """
        Args:
            q: Query tensor of shape (seqlen, d_head)
            k: Key tensor of shape (seqlen, d_head)
            v: Value tensor of shape (seqlen, d_head)
        
        Returns:
            attn_out: Attention output of shape (seqlen, d_head)
        """
        # Apply softmax scale
        softmax_scale = 0.125
        q_scaled = q * softmax_scale
        
        # Compute attention scores: Q @ K^T
        raw_score = torch.matmul(q_scaled, k.transpose(1, 0))
        
        # Apply softmax
        norm_score = torch.nn.functional.softmax(raw_score, dim=-1)
        
        # Compute output: scores @ V
        attn_out = torch.matmul(norm_score, v)
        
        return attn_out

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
    d_head = 64  # Head dimension
    seqlen = 4096  # Query sequence length
    
    # Create input tensors with shapes matching numpy reference
    q = torch.randn([seqlen, d_head], dtype=torch.float32)
    k = torch.randn([seqlen, d_head], dtype=torch.float32)
    v = torch.randn([seqlen, d_head], dtype=torch.float32)
    
    # Create model
    model = Attention()
    model.eval()
    
    description = f'Attention: d_head={d_head}, seqlen={seqlen}'
    
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
