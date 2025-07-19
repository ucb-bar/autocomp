import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), followed by LogSumExp, LeakyReLU, 
    LeakyReLU, GELU, and GELU activations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        # Gemm
        x = self.linear(x)
        # LogSumExp
        x = torch.logsumexp(x, dim=1, keepdim=True)
        # LeakyReLU
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        # LeakyReLU
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        # GELU
        x = torch.nn.functional.gelu(x)
        # GELU
        x = torch.nn.functional.gelu(x)
        return x

batch_size = 128
in_features = 1024
out_features = 512

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]