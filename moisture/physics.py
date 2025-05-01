import torch

from vars import *

def S_e(h):
    return torch.where(
            h < 0, 
    (1 + (alpha*torch.abs(h))**n)**(-m),
            torch.ones_like(h) # when saturated
        )
def WRC(h):
    return theta_r + (theta_s - theta_r)*S_e(h)
def HC(h):
    S_e_h = S_e(h)
    K_s * S_e_h**(1/2) * (1 - (1 - S_e_h**(1/m))**m)**2
    pass
