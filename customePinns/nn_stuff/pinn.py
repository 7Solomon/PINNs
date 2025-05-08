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
        self.activation = nn.Sigmoid()  # important sonst nans bei log
        layer_list = []
        for i in range(len(layers_body) - 1):
            layer_list.append(nn.Linear(layers_body[i], layers_body[i+1]))
            if i < len(layers_body) - 2:
                layer_list.append(self.activation)
        self.body = nn.Sequential(*layer_list)

        layer_list = []
        for i in range(len(layers_head) - 1):
            layer_list.append(nn.Linear(layers_head[i], layers_head[i+1]))
            if i < len(layers_head) - 2:
                layer_list.append(self.activation)
        self.head_1 = nn.Sequential(*layer_list)
        self.head_2 = nn.Sequential(*layer_list)



    def forward(self, x):
        body_out = self.body(x)
        out_1 = self.head_1(torch.log(body_out) + 1e-10)  # log 0 no godto
        out_2 = self.head_2(torch.log(body_out) + 1e-10)  # so i do 1e-10 EPSILON poggers
        return body_out, out_1, out_2