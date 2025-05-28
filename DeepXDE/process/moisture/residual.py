from process.moisture.scale import scale_h, rescale_h
import torch
import deepxde as dde
from config import richards1DConfig

torch.autograd.set_detect_anomaly(True)

def smooth_step(x, eps=1e-5):
    return torch.sigmoid(x/eps)
# Van Genuchten stuff   
def S_e(h):
    """
    Effective saturation (VG)"""
    smooth = smooth_step(-h) # ca 1 for h < 0, ca 0 for h > 0
    #core = (1 + (scale_h(richards1DConfig.alpha)*torch.abs(h))**richards1DConfig.n)**(-richards1DConfig.m)   # scaled h is givin zero values which lead to linear pred 
    
    # Unscaleing h
    h_actual = rescale_h(h)
    core_exponent_base = richards1DConfig.alpha * torch.abs(h_actual)
    core = (1 + core_exponent_base**richards1DConfig.n)**(-richards1DConfig.m)
    
    S_e_h = smooth *core + (1-smooth)
    return torch.clamp(S_e_h, min=1e-7, max=1.0 - 1e-7)  # to get arround hard 0,1. so that loss nan doesnt happen 

def WRC(h):
    """
    Water retention Curve (VG)
    """
    return richards1DConfig.theta_r + (richards1DConfig.theta_s - richards1DConfig.theta_r)*S_e(h)
def HC(h):
    """
    Hydraulic Conductivity (VG)
    """
    S_e_h = S_e(h)
    return (richards1DConfig.K_s * S_e_h**(1/2) * (1 - (1 - S_e_h**(1/richards1DConfig.m))**richards1DConfig.m)**2) / richards1DConfig.K_s  # for scaling

def C(h):
    """
    Specific Moisture capacity (VG)
    """
    theta = WRC(h)
    theta_h = dde.grad.jacobian(theta, h, i=0, j=0) # dtheta/dh
    return theta_h

    

def residual_1d_mixed(x, y): # y is u_pred , x is [x,y,t]   # ohne gravitation   # jetzt 1d asl [z,t]
    theta = WRC(y)
    K = HC(y)
    
    theta_t = dde.grad.jacobian(theta,x,i=0,j=1)
    u_x = dde.grad.jacobian(y,x,i=0,j=1)
    
    Ku_x = K * u_x
    Ku_xx = dde.grad.jacobian(Ku_x,x,i=0,j=1)

    return theta_t - Ku_xx

#def residual_1d_head(x, y):
#     = C(y)