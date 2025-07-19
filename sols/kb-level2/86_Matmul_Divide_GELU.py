import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, divides by a scalar, and applies GELU activation.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        x = self.linear(x)
        x = x / self.divisor
        x = torch.nn.functional.gelu(x)
        return x

batch_size = 128
input_size = 512
output_size = 1024
divisor = 10.0

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, output_size, divisor]