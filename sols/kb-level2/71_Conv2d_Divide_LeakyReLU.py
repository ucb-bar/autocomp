import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, divides by a constant, and applies LeakyReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor

    def forward(self, x):
        x = self.conv(x)
        x = x / self.divisor
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
divisor = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]