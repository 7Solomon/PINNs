import torch
import deepxde as dde
from config import richards1DConfig, concreteData

torch.autograd.set_detect_anomaly(True)

def smooth_step(x, eps=1e-5):
    return torch.sigmoid(x/eps)

# Van Genuchten stuff   
#def S_e_scaled(h):
#    """
#    Effective saturation (VG)"""
#    smooth = smooth_step(-h) # ca 1 for h < 0, ca 0 for h > 0
#    #core = (1 + (scale_h(richards1DConfig.alpha)*torch.abs(h))**richards1DConfig.n)**(-richards1DConfig.m)   # scaled h is givin zero values which lead to linear pred 
#    
#    # Unscaleing h
#    h_actual = rescale_h(h)
#    core_exponent_base = richards1DConfig.alpha * torch.abs(h_actual)
#    core = (1 + core_exponent_base**richards1DConfig.n)**(-richards1DConfig.m)
#    
#    S_e_h = smooth *core + (1-smooth)
#    return torch.clamp(S_e_h, min=1e-7, max=1.0 - 1e-7)  # to get arround hard 0,1. so that loss nan doesnt happen 

#def WRC_scaled(h):
#    """
#    Water retention Curve (VG)
#    """
#    return richards1DConfig.theta_r + (richards1DConfig.theta_s - richards1DConfig.theta_r)*S_e_scaled(h)
#def HC_scaled(h):
#    """
#    Hydraulic Conductivity (VG)
#    """
#    S_e_h = S_e_scaled(h)
#    return (richards1DConfig.K_s * S_e_h**(1/2) * (1 - (1 - S_e_h**(1/richards1DConfig.m))**richards1DConfig.m)**2) / richards1DConfig.K_s  # for scaling


#def residual_1d_mixed(x, y): # y is u_pred , x is [x,y,t]   # ohne gravitation   # jetzt 1d asl [z,t]
#    theta = WRC_scaled(y)
#    K = HC_scaled(y)
#
#    theta_t = dde.grad.jacobian(theta,x,i=0,j=1)
#    u_x = dde.grad.jacobian(y,x,i=0,j=1)
#    
#    Ku_x = K * u_x
#    Ku_xx = dde.grad.jacobian(Ku_x,x,i=0,j=1)
#
#    return theta_t - Ku_xx
#
#def residual_1d_head(x, y):
#    C = SMC(y) # C(h)
#    dh_dt = dde.grad.jacobian(y,x,i=0,j=1)  # dh/dt
#
#    K = HC_scaled(y) # K(h)
#    dh_dz = dde.grad.jacobian(y,x,i=0,j=0) # dh/dz
#
#    Kdh_dz = K * dh_dz
#    Kdh_dz_z = dde.grad.jacobian(Kdh_dz,x,i=0,j=0)  # d/dz(K * dh/dz)
#    return C * dh_dt - Kdh_dz_z  # - Sinkterm if i want 
#
#def theta(h):
#    core = (1 + (richards1DConfig.alpha * torch.abs(h))**richards1DConfig.n)**(-richards1DConfig.m)
#    return richards1DConfig.theta_r + (richards1DConfig.theta_s - richards1DConfig.theta_r) * core
#
#def S_e(h):
#    """
#    Effective saturation (VG)
#    """
#    smooth = smooth_step(-h) # ca 1 for h < 0, ca 0 for h > 0
#  
#    core_exponent_base = richards1DConfig.alpha * torch.abs(h)
#    core = (1 + core_exponent_base**richards1DConfig.n)**(-richards1DConfig.m)
#
#    S_e_h = smooth *core + (1-smooth)
#    return torch.clamp(S_e_h, min=1e-7, max=1.0 - 1e-7)  # to get arround hard 0,1. so that loss nan doesnt happen
#
#def WRC(h):
#    """
#    Water retention Curve / theta
#    """
#    return richards1DConfig.theta_r + (richards1DConfig.theta_s - richards1DConfig.theta_r) *   (h)
#def HC(h):
#    """
#    Hydraulic Conductivity / K
#    """
#    S_e_h = S_e(h)
#    return (richards1DConfig.K_s * S_e_h**(1/2) * (1 - (1 - S_e_h**(1/richards1DConfig.m))**richards1DConfig.m)**2)
#
#def SMC(theta):
#    """
#    Specific Moisture capacity / C
#    """
#    (richards1DConfig.theta_s - richards1DConfig.theta_r) * richards1DConfig.alpha * richards1DConfig.n * richards1DConfig.m *(richards1DConfig.alpha * torch.abs(theta))**(richards1DConfig.n - 1) / (1 + (richards1DConfig.alpha * torch.abs(theta))**richards1DConfig.n)**(richards1DConfig.m + 2)
#def SWD(theta):
#    """
#    Soil water diffusivity / D
#    """
#    C = SMC(theta)  # C(theta)
#    C = torch.clamp(C, min=1e-10)  # to avoid division by zero
#    Ktheta = HC(theta)/C  # K(theta)/C(theta)
#    return Ktheta
#
#def residual_1d_saturation(x,y):
#
#    dtheta_dt = dde.grad.jacobian(y,x,i=0,j=1)  # dtheta/dt
#
#    Dtheta = SWD(y)  # D(theta)
#    right_side = dde.grad.jacobian(Dtheta,x,i=0,j=0)  # (dtheta/dz)*(D(theta))
#    grad = dde.grad.jacobian(right_side,x,i=0,j=0)  # d/dz(D(theta)*dtheta/dz)
#    return grad - dtheta_dt  # - S




### Darcy
def residual_2d_darcy(x, y):
    """
    Darcy's law for 2D flow
    """
    K = concreteData.K
    d2h_dx2 = dde.grad.hessian(y, x, i=0, j=0)  # dh/dx
    d2h_dy2 = dde.grad.hessian(y, x, i=0, j=1)  # dh/dy
    return -K * (d2h_dx2 + d2h_dy2)  # -K * grad(h)