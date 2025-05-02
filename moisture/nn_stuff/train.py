from tqdm import tqdm
from nn_stuff.moist_pinn import residual
from utils import ConditionType, Domain
import torch
import torch.nn as nn

from vars import *

def train_loop(model, optimizer, mse_loss, domain: Domain, epochs):
    Loss = []
    #print(f'req_grad: {domain_collocation_tensor.requires_grad}')
    for epoch in tqdm(range(epochs), desc= 'Training ist im vollen gange'):
        optimizer.zero_grad()
        
        # Init loss
        loss_init = 0
        inital_points = domain.initial_condition.points.detach()
        pred = model(inital_points)
        loss_init = mse_loss(pred, domain.initial_condition.values)

        # Boundary loss
        boundary_loss = []
        for key in domain.condition_keys:
            current_points = domain.conditions[key].points.detach().requires_grad_(True)
            pred = model(current_points)
            if domain.conditions[key].type == ConditionType.DIRICHTLETT:
                temp_condition_loss = mse_loss(pred,domain.conditions[key].values)
            elif domain.conditions[key].type == ConditionType.NEUMANN:
                grad_pred = torch.autograd.grad(pred, current_points, grad_outputs=torch.ones_like(pred), create_graph=True)[0]
                # Verbose but okay for now
                if domain.conditions[key].key == 'left' or domain.conditions[key].key == 'right':
                    dh_dn = grad_pred[:, 0]
                elif domain.conditions[key].key == 'top' or domain.conditions[key].key == 'bottom':
                    dh_dn = grad_pred[:, 1]
                else:
                    raise ValueError(f'Condition key ist weird: {domain.conditions[key].type}')
                temp_condition_loss = mse_loss(dh_dn, domain.conditions[key].values)
            else:
                raise ValueError(f'Condition type  ist weird: {domain.conditions[key].type}')
            boundary_loss.append(temp_condition_loss)
        boundary_loss = sum([_ for _ in boundary_loss])
        # Residual loss
        coll_points = domain.collocation.detach().requires_grad_(True)
        res = residual(model, coll_points)
        loss_pde = mse_loss(res, torch.zeros_like(res))

        loss = lambda_bc * boundary_loss + lambda_pde * loss_pde + lambda_ic * loss_init
        # ---
        if epoch % 100 == 0:
            print(f'epoch: {epoch}')
            print(f'loss_pde: {loss_pde}, scaled: {loss_pde*lambda_pde}')
            print(f'loss_init: {loss_init}, scaled: {loss_init*lambda_ic}')
            print(f'boundary_loss: {boundary_loss}, scaled: {boundary_loss*lambda_bc}')

            print('---')
            print(f'loss: {loss}')
        if torch.isnan(loss):
            print('Loss ist NaN')
            break
        Loss.append(loss.item())
        # --
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_NORM)
        optimizer.step()
    return {
        'model': model,
        'loss': Loss,
    }

#print(model(domain_collocation_tensor).detach().cpu().numpy().min())
#print(model(domain_collocation_tensor).detach().cpu().numpy().max())

