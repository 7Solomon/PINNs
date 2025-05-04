import torch
from moisture.vars import *

def smooth_step(x, eps=1e-5):
    return 1 / (1 + torch.exp(-x/eps))

# Van Genuchten stuff   
def S_e(h):
    """
    Effective saturation (VG)"""
    smooth = smooth_step(-h) # ca 1 for h < 0, ca 0 for h > 0
    core = (1 + (alpha*torch.abs(h))**n)**(-m)
    return smooth *core + (1-smooth)
    #return torch.where(    # Not good because gradient jump
    #        h < 0, 
    #(1 + (alpha*torch.abs(h))**n)**(-m),
    #        torch.ones_like(h) # when saturated
    #    )
def WRC(h):
    """
    Water retention Curve (VG)
    """
    return theta_r + (theta_s - theta_r)*S_e(h)
def HC(h):
    """
    Hydraulic Conductivity (VG)
    """
    S_e_h = S_e(h)
    return K_s * S_e_h**(1/2) * (1 - (1 - S_e_h**(1/m))**m)**2
    
