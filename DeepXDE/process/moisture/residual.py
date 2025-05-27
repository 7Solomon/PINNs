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
    

def residual(x, y): # y is u_pred , x is [x,y,t]   # ohne gravitation   # jetzt 1d
    theta = WRC(y)
    K = HC(y)
    
    theta_t = dde.grad.jacobian(theta,x,i=0,j=1)
    grad_u = dde.grad.jacobian(y,x,i=0,j=None)

    u_x = grad_u[:,0]
    #u_y = grad_u[:,1]

    Ku_x = K * u_x
    #Ku_y = K * u_y

    Ku_xx = dde.grad.jacobian(Ku_x,x,i=0,j=1)
    #Ku_yy = dde.grad.jacobian(Ku_y,x,i=0,j=1)

    #return theta_t - (Ku_xx + Ku_yy)
    return theta_t - Ku_xx


#def residual(x, y): # y is h_scaled (network output), x is [z_scaled, t_scaled]  ##### GEVIBBED
#    # Scaling constants based on your scale.py
#    Z_DIVISOR = 1.0
#    T_DIVISOR = 1.1e10
#    H_DIVISOR = 1000.0
#
#    # theta is theta_actual(h_actual(y)) because WRC uses S_e which uses rescale_h(y)
#    theta = WRC(y)
#    # K_r is K_r_actual(h_actual(y)) because HC uses S_e and divides by K_s
#    K_r = HC(y) 
#    
#    # d(theta_actual)/d(t_scaled)
#    # x[:, 1] is t_scaled
#    theta_t = dde.grad.jacobian(theta, x, i=0, j=1)
#    
#    # grad_y is [d(h_scaled)/d(z_scaled), d(h_scaled)/d(t_scaled)]
#    # x[:, 0] is z_scaled
#    grad_y = dde.grad.jacobian(y, x, i=0, j=None) 
#    # h_scaled_z_scaled = d(h_scaled)/d(z_scaled)
#    h_scaled_z_scaled = grad_y[:, 0:1] # Shape (N, 1) to be robust for multiplication
#
#    # This is K_r_actual * d(h_scaled)/d(z_scaled)
#    K_r_h_scaled_z_scaled = K_r * h_scaled_z_scaled 
#    
#    # This is d/d(z_scaled) [ K_r_actual * d(h_scaled)/d(z_scaled) ]
#    div_flux_scaled = dde.grad.jacobian(K_r_h_scaled_z_scaled, x, i=0, j=0)
#
#    # Prefactor for the scaled PDE
#    PREFACTOR = (T_DIVISOR * richards1DConfig.K_s * H_DIVISOR) / (Z_DIVISOR**2)
#
#    # PDE residual: 
#    # d(theta_actual)/dt_scaled - PREFACTOR * d/dz_scaled (K_r_actual * dh_scaled/dz_scaled) = 0
#    pde_res = theta_t - PREFACTOR * div_flux_scaled
#    
#    return pde_res