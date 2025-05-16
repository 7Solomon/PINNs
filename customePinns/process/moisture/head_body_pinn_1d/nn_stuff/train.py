
from tqdm import tqdm
from moisture.head_body_pinn_1d.residual import predicted_K_theta_residual
from utils import Domain
import torch
import torch.nn as nn
from nn_stuff.pinn import BodyHeadPINN

def train_body_head_loop(model: BodyHeadPINN,optimizer ,domain: Domain,conf ):
    Loss = []
    # scale conditions
    domain.scale_conditions()

    for epoch in tqdm(range(conf.epochs), desc= 'Training ist im vollen gange'):
        optimizer.zero_grad()

        # Residual loss
        coll_points = domain.collocation.detach().requires_grad_(True)
        
        res = predicted_K_theta_residual(model, coll_points, epoch % 100 == 0)
        loss_pde = conf.mse_loss(res, torch.zeros_like(res))
        loss = conf.lambda_pde * loss_pde
        # ---
        #if epoch % 100 == 0:
        #    print(f'epoch: {epoch}')
#
        #    print('---')
        #    print(f'loss: {loss}')
        if torch.isnan(loss):
            print('Loss ist NaN')
            break
        Loss.append(loss.item())
        # --
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=conf.max_norm)
        optimizer.step()
    return {
        'model': model,
        'loss': Loss,
    }

#print(model(domain_collocation_tensor).detach().cpu().numpy().min())
#print(model(domain_collocation_tensor).detach().cpu().numpy().max())

