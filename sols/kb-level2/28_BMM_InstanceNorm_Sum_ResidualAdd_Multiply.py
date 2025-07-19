import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Model that performs a batch matrix multiplication, instance normalization, summation, residual addition, and multiplication.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Input tensor of shape (batch_size, out_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.bmm(x)
        x = self.instance_norm(x.unsqueeze(1).unsqueeze(1)).squeeze(1).squeeze(1)
        x = x + y
        x = x * y
        return x

batch_size = 128
in_features = 64
out_features = 128

def get_inputs():
    return [torch.randn(batch_size, in_features), torch.randn(batch_size, out_features)]

def get_init_inputs():
    return [in_features, out_features]