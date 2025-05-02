from physics import *

class MoisturePINN(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = torch.nn.Tanh()  
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(torch.nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                layer_list.append(self.activation)
        self.model = torch.nn.Sequential(*layer_list)

    def forward(self, x):
        return self.model(x)

def residual(model, x):
    h = model(x)

    theta = WRC(h)
    K = HC(h)
    
    theta_t = torch.autograd.grad(theta, x, grad_outputs=torch.ones_like(theta), create_graph=True)[0][:,2]

    grad_h = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(theta), create_graph=True)[0]  # Alle Gradienten nach x, y, t
    h_x = grad_h[:, 0]
    h_y = grad_h[:, 1]

    K_h_x = K  * h_x
    K_h_y = K  * h_y

    d_Kh_dx = torch.autograd.grad(K_h_x, x, grad_outputs=torch.ones_like(K_h_x), create_graph=True)[0][:, 0]
    d_Kh_dy = torch.autograd.grad(K_h_y, x, grad_outputs=torch.ones_like(K_h_y), create_graph=True)[0][:, 1]
    
    div_K_grad_h = d_Kh_dx + d_Kh_dy
    return theta_t - div_K_grad_h