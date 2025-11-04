import torch
import torch.nn as nn
import logging
import benchmark_lib

# Setup logger (will be configured by benchmark_lib)
logger = logging.getLogger(__name__)


# ============================================================================
# MODULE DEFINITION - Replace this section to benchmark different modules
# ============================================================================

def associative_scan(deltaA, deltaB_u):
    """
    Args:
        deltaA: [batch_size, channels, state_size, seq_len]
        deltaB_u: [batch_size, channels, state_size, seq_len]

    Mamba uses an associative scan operator to aggregate information across
    time sequentially (sequence length, e.g. sequence of tokens),
    from the past to the present.
    """
    batch_size, channels, state_size, seq_len = deltaA.shape
    out = torch.empty(batch_size, channels, state_size, seq_len,
                        device=deltaA.device, dtype=deltaA.dtype)
    for i in range(seq_len):
        prev_state = out[..., i - 1] if i > 0 else 0
        out[..., i] = deltaA[..., i] * prev_state + deltaB_u[..., i]
    return out

def mamba_layer(delta, A, B, u, C):
    """
    Args:
        delta: [batch, channels, seq_len]
        u: [batch, channels, seq_len]
        A: [channels, state_size]
        B: [batch, state_size, seq_len]
        C: [batch, state_size, seq_len]
    """
    # expand the tensors so they all have the same dimensions and compute elementwise products (with broadcast)
    # deltaA and deltaB_u have shape [batch_size, channels, state_size, seq_len]
    deltaA = torch.exp(delta[:, :, None, :] * A[None, :, :, None])
    deltaB_u = delta[:, :, None, :] * B[:, None, :, :] * u[:, :, None, :]
    scan_res = associative_scan(deltaA, deltaB_u)
    # y sums over the `state_size` axis and has shape [batch_size, channels, seq_len]
    mamba_out = (C[:, None, :, :] * scan_res).sum(dim=-2)
    return mamba_out

@torch.no_grad()
class MambaLayer(nn.Module):
    """MambaLayer module using custom mamba_layer function"""
    def __init__(self):
        super().__init__()

    def forward(self, delta, A, B, u, C):
        """
        Args:
            delta: [batch, channels, seq_len]
            A: [channels, state_size]
            B: [batch, state_size, seq_len]
            u: [batch, channels, seq_len]
            C: [batch, state_size, seq_len]
        """
        return mamba_layer(delta, A, B, u, C)


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
    batch_size = 1
    channels = 256
    state_size = 16
    seq_len = 2048
    
    # Create input tensors for MambaLayer
    # delta: [batch, channels, seq_len]
    delta = torch.randn([batch_size, channels, seq_len], dtype=torch.float32)
    # A: [channels, state_size]
    A = torch.randn([channels, state_size], dtype=torch.float32)
    # B: [batch, state_size, seq_len]
    B = torch.randn([batch_size, state_size, seq_len], dtype=torch.float32)
    # u: [batch, channels, seq_len]
    u = torch.randn([batch_size, channels, seq_len], dtype=torch.float32)
    # C: [batch, state_size, seq_len]
    C = torch.randn([batch_size, state_size, seq_len], dtype=torch.float32)
    
    # Create MambaLayer model
    model = MambaLayer()
    model.eval()
    
    description = f'MambaLayer: batch={batch_size}, channels={channels}, state_size={state_size}, seq_len={seq_len}'
    
    # Return inputs as a tuple (delta, A, B, u, C)
    return model, (delta, A, B, u, C), description

# ============================================================================
# END MODULE DEFINITION
# ============================================================================


if __name__ == "__main__":
    # Create model and inputs
    model, inputs, description = create_model_and_input()
    
    # Run benchmark
    benchmark_lib.run_benchmark(model, inputs, description)
