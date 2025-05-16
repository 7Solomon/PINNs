from utils import ConditionType
import torch
import torch.nn as nn
from tqdm import tqdm

from mechanic.residual import residual
def train_bernouli_balken_loop(model, optimizer, domain, conf):
    Loss = []
    model.train()
    
    pBar = tqdm(range(conf.epochs), desc='Training ist im vollen gange')
    for epoch in pBar:
        optimizer.zero_grad()
        #print(domain.collocation[:2])
        pred_domain = model(domain.collocation)
        
        residal = residual(pred_domain, conf, domain.collocation)
        residual_loss = nn.MSELoss()(residal, torch.zeros_like(residal))

        boundary_loss = force_boundary_conditions(model, domain, conf)

        loss = conf.lambda_pde * residual_loss + conf.lambda_bc * boundary_loss

        if torch.isnan(loss):
            print('IST NAN')
            break

        Loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            pBar.set_postfix(loss=f'{loss.item():.4e}')
    return {
        'model': model,
        'loss': Loss,
    }

def force_boundary_conditions(mode, domain, conf):
    # Dirichlet
    for key, value in domain.boundarys.items():
        if domain.boundarys[key].type  == ConditionType.DIRICHLET:
            #print('domain.boundarys[key].points ', domain.boundarys[key].points[:2])
            #print('domain.boundarys[key].values ', domain.boundarys[key].values[:2])
            return nn.MSELoss()(domain.boundarys[key].points, domain.boundarys[key].values)
        elif domain.boundarys[key].type == ConditionType.NEUMANN:
            raise NotImplementedError('Neumann conditions are not implemented yet, you silly boy')
            #return torch.MSELoss()(, domain.conditions[key].values)

        
