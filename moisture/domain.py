import torch
import numpy as np

def get_domain(n_points=1000, x_max=1, y_max=2, t_max= 10):
    x = torch.rand(n_points, 1, requires_grad=True) * x_max
    y = torch.rand(n_points, 1, requires_grad=True) * y_max
    t = torch.rand(n_points, 1, requires_grad=True) * t_max
    return torch.cat([x, y, t], dim=1)  # [1,3] tensor

def get_boundary():
    pass
