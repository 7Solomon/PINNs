import torch
from config import concreteData

def output_transform(x, y):
    return torch.sigmoid(y) * (concreteData.theta_s/0.35)