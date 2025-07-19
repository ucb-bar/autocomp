import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies ReLU, and divides by a constant.
    """
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        x = x / self.divisor
        return x

batch_size = 128
in_features = 1024
out_features = 512
divisor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, divisor]