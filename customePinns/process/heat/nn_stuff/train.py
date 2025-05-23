from tqdm import tqdm
from utils import Domain

from heat.nn_stuff.physics import steady_lp_residual
import torch
import torch.nn as nn

from heat.vars import *

def train_steady_loop(model, optimizer, domain: Domain, conf):
    Loss = [] 
    boundary_Loss, pde_Loss = [],[]
    boundary_Loss_scaled, pde_Loss_scaled = [],[]
 
    # scale conditions
    print(domain.condition_keys)
    domain.scale_conditions()

    #print(f'req_grad: {domain_collocation_tensor.requires_grad}')
    for epoch in tqdm(range(conf.epochs), desc= 'Training ist im vollen gange'):
        optimizer.zero_grad()
        
        # Boundary loss
        boundary_loss = []
        for key in domain.condition_keys:
            pred = model(domain.conditions[key].points)
            boundary_loss.append(conf.mse_loss(pred, torch.tensor(domain.conditions[key].scaled_values, dtype=torch.float32)))
        boundary_loss = sum([_ for _ in boundary_loss])

        # Residual loss
        res = steady_lp_residual(model, domain.collocation)
        loss_pde = conf.mse_loss(res, torch.zeros_like(res))

        loss = conf.lambda_bc * boundary_loss + conf.lambda_pde * loss_pde
        Loss.append(loss.item())
        boundary_Loss.append(boundary_loss.item())
        pde_Loss.append(loss_pde.item())
        boundary_Loss_scaled.append(boundary_loss.item() * conf.lambda_bc)
        pde_Loss_scaled.append(loss_pde.item() * conf.lambda_pde)
        loss.backward()
        optimizer.step()
    return {
        'model': model,
        'loss': {
            'total':Loss,
            'boundary': {
                'true':boundary_Loss,
                'scaled': boundary_Loss_scaled
                },
            'residual': {
                'true':pde_Loss,
                'scaled': pde_Loss_scaled
                },
            },
    }

def train_transient_loop(model, optimizer, mse_loss, domain: Domain, epochs=500):
    Loss = []

    # scale conditions
    domain.scale_conditions()
    #print(f'req_grad: {domain_collocation_tensor.requires_grad}')
    for epoch in tqdm(range(epochs), desc= 'Training ist im vollen gange'):
        optimizer.zero_grad()
        
        # Init loss
        pred = model(domain.initial_condition.points)
        loss_init = mse_loss(pred, torch.tensor(domain.initial_condition.values, dtype=torch.float32))

        # Boundary loss
        boundary_loss = []
        for key in domain.condition_keys:
            pred = model(domain.conditions[key].points)
            boundary_loss.append(mse_loss(pred, torch.tensor(domain.conditions[key].values, dtype=torch.float32)))
        boundary_loss = sum([_ for _ in boundary_loss])
        
        # Residual loss
        res = steady_lp_residual(model, domain.collocation)
        loss_pde = mse_loss(res, torch.zeros_like(res))

        loss = lambda_bc * boundary_loss + lambda_pde * loss_pde + lambda_ic * loss_init
        Loss.append(loss.item())
        loss.backward()
        optimizer.step()
    return {
        'model': model,
        'loss': Loss,
    }


#print(model(domain_collocation_tensor).detach().cpu().numpy().min())
#print(model(domain_collocation_tensor).detach().cpu().numpy().max())

