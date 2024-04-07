import torch
import numpy as np

class ANN_net(torch.nn.Module):
    def __init__(self, layer_dim_list, activation_function):
        super().__init__()
        layers_list = torch.nn.ModuleList()
        for layer_num in range(0, len(layer_dim_list) - 1):
            layers_list.append(torch.nn.Linear(layer_dim_list[layer_num], layer_dim_list[layer_num + 1]))
        self.layers_list = layers_list
        self.activation_function = activation_function
    
    def forward(self, *args):
        x = torch.vstack((args[0])).T
        for i in range(len(self.layers_list)-1):
            x = self.activation_function(self.layers_list[i](x))
        result = self.layers_list[-1](x)
        return result
