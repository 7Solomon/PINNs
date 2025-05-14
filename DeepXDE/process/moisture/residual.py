import torch
import deepxde as dde
from config import richardsConfig

torch.autograd.set_detect_anomaly(True)

def smooth_step(x, eps=1e-5):
    #return 1 / (1 + torch.exp(-x/eps))
    return torch.sigmoid(x/eps)
# Van Genuchten stuff   
def S_e(h):
    """
    Effective saturation (VG)"""
    smooth = smooth_step(-h) # ca 1 for h < 0, ca 0 for h > 0
    core = (1 + (richardsConfig.alpha*torch.abs(h))**richardsConfig.n)**(-richardsConfig.m)
    S_e_h = smooth *core + (1-smooth)
    return torch.clamp(S_e_h, min=1e-7, max=1.0 - 1e-7)  # to get arround hard 0,1. so that loss nan doesnt happen 

def WRC(h):
    """
    Water retention Curve (VG)
    """
    return richardsConfig.theta_r + (richardsConfig.theta_s - richardsConfig.theta_r)*S_e(h)
def HC(h):
    """
    Hydraulic Conductivity (VG)
    """
    S_e_h = S_e(h)
    return richardsConfig.K_s * S_e_h**(1/2) * (1 - (1 - S_e_h**(1/richardsConfig.m))**richardsConfig.m)**2
    

def residual(x, y): # y is u_pred , x is [x,y,t]   # ohne gravitation
    theta = WRC(y)
    K = HC(y)
    
    theta_t = dde.grad.jacobian(theta,x,i=0,j=2)
    grad_u = dde.grad.jacobian(y,x,i=0)


    u_x = grad_u[:,0]
    u_y = grad_u[:,1]

    Ku_x = K * u_x
    Ku_y = K * u_y

    Ku_xx = dde.grad.jacobian(Ku_x,x,i=0,j=0)
    Ku_yy = dde.grad.jacobian(Ku_y,x,i=0,j=1)

    #print('theta_t', theta_t[:1,:])
    #print('Ku_xx', Ku_xx[:1,:])
    #print('Ku_yy', Ku_yy[:1,:])
    return theta_t - (Ku_xx + Ku_yy)

    