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
class LayerNormModule(nn.Module):
    """LayerNorm module using custom layernorm_layer function"""
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, input_tensor, gamma, beta):
        # Compute the mean and variance of the input tensor along the last dimension
        mean = input_tensor.mean(dim=-1, keepdim=True)
        variance = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
        # Subtract the mean from the input and divide by the square root of the variance plus epsilon
        normalized_input = (input_tensor - mean) / torch.sqrt(variance + self.epsilon)
        # Apply the affine transformation
        normalized_input = normalized_input * gamma + beta
        return normalized_input


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
    epsilon = 1e-5
    
    # Create input tensor with shape [B, seq_len, hidden_size]
    input_tensor = torch.randn([batch_size, seq_len, hidden_size], dtype=torch.float32)
    
    # Create gamma and beta tensors with shape [hidden_size]
    gamma = torch.randn([hidden_size], dtype=torch.float32)
    beta = torch.randn([hidden_size], dtype=torch.float32)
    
    # Create LayerNorm model
    model = LayerNormModule(epsilon)
    model.eval()
    
    description = f'LayerNorm: input=[{batch_size}, {seq_len}, {hidden_size}], normalized_shape={hidden_size}'
    
    # Return inputs as a tuple (input_tensor, gamma, beta)
    return model, (input_tensor, gamma, beta), description

# ============================================================================
# END MODULE DEFINITION
# ============================================================================


if __name__ == "__main__":
    # Create model and inputs
    model, inputs, description = create_model_and_input()
    
    # Run benchmark
    benchmark_lib.run_benchmark(model, inputs, description)
