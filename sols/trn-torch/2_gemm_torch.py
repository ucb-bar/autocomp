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
class Matmult(nn.Module):
    """Matrix multiplication module using nn.Linear"""
    def __init__(self, in_features, out_features, is_add_bias):
        super().__init__()
        self.matmult = nn.Linear(in_features=in_features, out_features=out_features, bias=is_add_bias, dtype=torch.bfloat16)

    def forward(self, x):
        return self.matmult(x).mean()


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
    m = 4096
    k = 8192
    n = 8192
    add_bias = True
    
    # Create input matrix with shape [B, M, K]
    input_tensor = torch.randn([batch_size, m, k], dtype=torch.bfloat16)
    
    # Create model with K input features and N output features
    model = Matmult(k, n, add_bias)
    model.eval()
    
    description = f'Matrix Multiplication: input=[{batch_size}, {m}, {k}], output=[{batch_size}, {m}, {n}]'
    
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
