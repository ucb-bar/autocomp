import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Model that performs a convolution, scales the output, and then applies a minimum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = self.conv(x)
        x = x * self.scale_factor
        x = torch.min(x, dim=1, keepdim=True)[0]  # Minimum along channel dimension
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]