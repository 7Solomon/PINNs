from tqdm import tqdm
from nn_stuff.physics import steady_lp_residual
from utils import Domain
import torch
import torch.nn as nn

from vars import *

def train_loop(model, optimizer, mse_loss, domain: Domain, epochs=500):
    Loss = []
    #print(f'req_grad: {domain_collocation_tensor.requires_grad}')
    for epoch in tqdm(range(epochs), desc= 'Training ist im vollen gange'):
        optimizer.zero_grad()
        res = steady_lp_residual(model, domain.collocation)
        
        boundary_loss = []
        for key in domain.keys:
            pred = model(domain.boundaries[key])
            boundary_loss.append(mse_loss(pred, torch.tensor(domain.values[key], dtype=torch.float32)))
        loss_pde = mse_loss(res, torch.zeros_like(res))

        loss = lambda_bc * sum([_ for _ in boundary_loss]) + lambda_pde * loss_pde
        Loss.append(loss.item())
        loss.backward()
        optimizer.step()
    return {
        'model': model,
        'loss': Loss,
    }

#print(model(domain_collocation_tensor).detach().cpu().numpy().min())
#print(model(domain_collocation_tensor).detach().cpu().numpy().max())

