import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies ReLU, and adds a bias term.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = x + self.bias
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]