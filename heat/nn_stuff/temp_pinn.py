import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()    # RELU NOT GOOD GIVES 0 second derivative
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                layer_list.append(self.activation)
        self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.model(x)
    
    
