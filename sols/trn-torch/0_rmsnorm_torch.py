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
class RMSNormModule(nn.Module):
    """RMSNorm module using torch_rmsnorm_kernel function"""
    def __init__(self):
        super().__init__()

    def forward(self, a_tensor, g_tensor):
        # Square the tensor (element-wise)
        in_square = a_tensor.pow(2)
        # Calculate means in the free dimension
        mean = in_square.mean(dim=1, keepdim=True)
        # Scale by reciprocal of sqrt(mean)
        tensor = a_tensor * torch.rsqrt(mean)
        
        # Scale the output by the weight
        return tensor * g_tensor


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
    batch_size = 4096
    hidden_size = 512
    
    # Create input tensor with shape [hidden_size, batch_size]
    a_tensor = torch.randn([hidden_size, batch_size], dtype=torch.bfloat16)
    g_tensor = torch.randn([batch_size], dtype=torch.bfloat16)
    
    # Create RMSNorm model
    model = RMSNormModule()
    model.eval()
    
    description = f'RMSNorm: input=[{hidden_size}, {batch_size}], normalized_shape={hidden_size}'
    
    # Return inputs as a tuple (input_tensor, gamma)
    return model, (a_tensor, g_tensor), description

# ============================================================================
# END MODULE DEFINITION
# ============================================================================


if __name__ == "__main__":
    # Create model and inputs
    model, inputs, description = create_model_and_input()
    
    # Run benchmark
    benchmark_lib.run_benchmark(model, inputs, description)
