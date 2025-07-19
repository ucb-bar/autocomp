import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Model that performs a convolution, applies Group Normalization, Tanh, HardSwish, 
    Residual Addition, and LogSumExp.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, x):
        # Convolution
        x_conv = self.conv(x)
        # Group Normalization
        x_norm = self.group_norm(x_conv)
        # Tanh
        x_tanh = self.tanh(x_norm)
        # HardSwish
        x_hard_swish = self.hard_swish(x_tanh)
        # Residual Addition
        x_res = x_conv + x_hard_swish
        # LogSumExp
        x_logsumexp = torch.logsumexp(x_res, dim=1, keepdim=True)
        return x_logsumexp

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
groups = 8

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups]