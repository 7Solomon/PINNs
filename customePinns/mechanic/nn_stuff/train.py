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
        loss = nn.MSELoss()(residal, torch.zeros_like(residal))

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
    for key, value in domain.condition.items():
        if domain.conditions[key].type  == ConditionType.DIRICHLET:
            return torch.MSELoss()(domain.conditions[key].points, domain.conditions[key].values)
        elif domain.conditions[key].type == ConditionType.NEUMANN:
            raise NotImplementedError('Neumann conditions are not implemented yet, you silly boy')
            #
            #return torch.MSELoss()(, domain.conditions[key].values)

    # Neumann
    for key in domain.condition_keys:
        i
    return pred

        
