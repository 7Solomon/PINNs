from pyexpat import model
import torch
import torch.nn as nn

alpha =7.5e-7  # m^2/s bissle rumspielen

T_max = 100.0
T_min = 0.0

layers = [2, 64, 64, 64, 1]
lambda_ic = 1.0 # W IC loss
lambda_bc = 1.0 # W bc loss
lambda_pde = 1.0 # W PDE residual loss
mse_loss = nn.MSELoss()
lr = 1e-3
EPOCHS = 300

MODEL_PATH = 'models'

