import torch
def output_transform_2d(x, y, scale):
    return torch.sigmoid(y) * 1 #(concreteData.theta_s/0.35)

