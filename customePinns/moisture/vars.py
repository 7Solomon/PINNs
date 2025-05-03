import torch.nn as nn
import torch

## Van genuchten
# WRC
theta_r = 0.0
theta_s = 0.1
alpha = 0.4 # ? ka was hier gut ist
n = 1.2 
m = 1 - (1/n)
K_s = 10e-9 # Niedrig da beton fast kein wasser leitet

# PINN
layers = [3, 64, 64, 64, 1]
lambda_ic = 1.0 # W IC loss
lambda_bc = 1.0 # W bc loss
lambda_pde = 1e9 # W PDE residual loss
mse_loss = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 5e-6
EPOCHS = 20000
MAX_NORM = 1.0

MODEL_PATH = 'models'

## will anders lösen
h_max = 60.0
h_min = 0.0 # m