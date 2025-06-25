import torch

def output_transform_1d_head(x, y, scale):
    return - torch.sigmoid(y) * (10/scale.h_char)

def output_transform_1d_saturation(x, y, scale):
    return torch.sigmoid(y) * 1 #(concreteData.theta_s/0.35)


#### WEGEN SCALING IMMER ZWISCHEN 0-1, aber head -