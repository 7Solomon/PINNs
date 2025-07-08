from torch import nn
import torch
import deepxde as dde

class MBNN(dde.maps.pytorch.NN):
    """
    A general Multi-Branch Neural Network (MBNN) for multiphysics problems.
    It can have an arbitrary number of branches, each processing a different
    subset of the input features.

    The outputs of all branches are concatenated to form the final output tensor.
    """
    def __init__(self, branch_definitions, activation=nn.Tanh()):
        """
        Args:
            branch_definitions (list of tuples): Each tuple defines a branch
                and should be in the format (input_indices, layer_dims).
                - input_indices (list of int): The indices of the input tensor `x`
                  that this branch will process.
                - layer_dims (list of int): List of layer sizes for the branch.
                  Example: [2, 20, 20, 1] (2 inputs, 2 hidden, 1 output)
            activation: The activation function to use (e.g., nn.Tanh()).
        
        Example for the original behavior:
        branch_definitions = [
            ([0, 1], [2, 20, 20, 2]), # uv_branch
            ([2],    [1, 20, 20, 1])  # t_branch
        ]
        """
        super().__init__()
        
        self.branches = nn.ModuleList()
        self.input_indices = []

        for indices, dims in branch_definitions:
            self.input_indices.append(indices)
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:  # No activation on the final output layer
                    layers.append(activation)
            self.branches.append(nn.Sequential(*layers))

    def forward(self, x):
        branch_outputs = []
        for i, branch in enumerate(self.branches):
            branch_input = x[:, self.input_indices[i]]
            branch_outputs.append(branch(branch_input))
        
        # Concatenate outputs
        return torch.cat(branch_outputs, dim=1)