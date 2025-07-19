import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        
        layers = []
        current_input_size = input_size
        
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            current_input_size = hidden_size
        
        layers.append(nn.Linear(current_input_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        return self.network(x)

# Test code
batch_size = 1
input_size = 1000
hidden_layer_sizes = [50, 50, 50, 50, 50, 50, 50, 50]  # Example of deep and narrow layers
output_size = 10

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]