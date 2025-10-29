import torch
import torch.nn as nn
import logging
import benchmark_lib

# Setup logger (will be configured by benchmark_lib)
logger = logging.getLogger(__name__)


# ============================================================================
# MODULE DEFINITION - Replace this section to benchmark different modules
# ============================================================================

@torch.no_grad()
class RMSNorm(nn.Module):
    """RMSNorm module"""
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, a_tensor):
        # Square the tensor (element-wise)
        in_square = a_tensor.pow(2)
        # Calculate means in the free dimension
        mean = in_square.mean(dim=1, keepdim=True)
        # Scale by reciprocal of sqrt(mean)
        tensor = a_tensor * torch.rsqrt(mean)
        # Scale the output by the weight
        return (tensor * self.weight).mean()


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
    seq_len = 4096
    hidden_size = 8192
    
    # Create input tensor with shape [batch_size, seq_len, hidden_size]
    input_tensor = torch.randn([batch_size, seq_len, hidden_size], dtype=torch.float32)
    
    # Create RMSNorm model
    model = RMSNorm(hidden_size)
    model.eval()
    
    description = f'RMSNorm: input=[{batch_size}, {seq_len}, {hidden_size}]'
    
    # Return inputs as a tuple (single input in this case)
    return model, (input_tensor,), description

# ============================================================================
# END MODULE DEFINITION
# ============================================================================


if __name__ == "__main__":
    # Create model and inputs
    model, inputs, description = create_model_and_input()
    
    # Run benchmark
    benchmark_lib.run_benchmark(model, inputs, description)
