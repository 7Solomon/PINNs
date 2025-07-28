import torch
import torch.nn as nn
import deepxde as dde

class MultiHeadNN(dde.maps.pytorch.NN):
    """
    A Multi-Head Neural Network for multiphysics problems.

    This network takes a single, complete input tensor (e.g., spatial and 
    temporal coordinates) and processes it through multiple independent 
    network "heads". Each head is responsible for predicting a different 
    physical quantity. The outputs of all heads are then concatenated.

    This is ideal for problems where different fields (e.g., velocity, pressure)
    are coupled but may benefit from having separate network parameters.
    """
    def __init__(self, input_dim, head_definitions, activation='tanh', initializer='Glorot uniform'):
        """
        Args:
            input_dim (int): The dimension of the input tensor (e.g., 3 for (x, y, t)).
            head_definitions (list of lists): Each inner list defines the
                layer dimensions for one output head, starting from the first
                hidden layer to the output layer of that head.
            activation: The activation function for hidden layers.
            initializer: The weight initializer for the network.
        Example:
            # For a problem with inputs (x, y, t) and outputs (u, v, p)
            input_dim = 3
            head_definitions = [
                [32, 32, 1],  # Head for 'u', with 2 hidden layers of 32 neurons
                [32, 32, 1],  # Head for 'v'
                [32, 32, 1]   # Head for 'p'
            ]
            net = MultiHeadNN(input_dim, head_definitions)
        """
        super().__init__()
        self.heads = nn.ModuleList()
        self.activation = getattr(nn, activation)() if isinstance(activation, str) else activation

        for head_dims in head_definitions:
            full_dims = [input_dim] + head_dims 
            layers = []
            for i in range(len(full_dims) - 1):
                linear = nn.Linear(full_dims[i], full_dims[i+1])
                if initializer == 'Glorot uniform':
                    nn.init.xavier_uniform_(linear.weight)
                    nn.init.zeros_(linear.bias)
                else:
                    raise ValueError(f"Unsupported initializer: {initializer}")
                layers.append(linear)
                if i < len(full_dims) - 2:
                    layers.append(self.activation)
            self.heads.append(nn.Sequential(*layers))

    def forward(self, x):
        """
        The input `x` is fed to all heads simultaneously.
        """
        head_outputs = []
        for head in self.heads:
            head_outputs.append(head(x))
        
        # Concatenate the outputs from all heads along the feature dimension
        return torch.cat(head_outputs, dim=1)