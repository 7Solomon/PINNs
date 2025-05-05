import torch.nn as nn
import torch

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
    
class BodyHeadPINN(nn.Module):
    def __init__(self, layers_body, layers_head):
        super().__init__()
        self.body = PINN(layers_body)
        self.head_1 = PINN(layers_head)
        self.head_2 = PINN(layers_head)

    def forward(self, x):
        body_out = self.body(x)
        out_1 = self.head_1(torch.log(body_out))
        out_2 = self.head_2(torch.log(body_out))
        return body_out, out_1, out_2