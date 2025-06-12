import torch
from config import concreteData

def output_transform_1d_head(x, y):
    return - torch.sigmoid(y) * 1#(10/10)

def output_transform_1d_saturation(x, y):
    return torch.sigmoid(y) * 1 #(concreteData.theta_s/0.35)


#### WEGEN SCALING IMMER ZWISCHEN 0-1, aber head -