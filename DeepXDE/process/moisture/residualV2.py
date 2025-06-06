from process.moisture.scale import Scale
#from config import richards1DConfig
import torch
import deepxde as dde

from config import concreteData

def S_e(h):
    """
    Effective saturation (VG)
    """
    core = (1 + (concreteData.soil_water_retention * torch.abs(h))**concreteData.n)**(-concreteData.m)
    
    smooth = torch.sigmoid(-h)  # ca 1 for h < 0, ca 0 for h > 0
    S_e = smooth * core + (1 - smooth)  # diffrentiable switch
    return torch.clamp(S_e, min=1e-7, max=1.0 - 1e-7) 


def water_retention_curve(h):
    """
    Computes the volumetric water content (VWC) out of head.
    """
    return concreteData.theta_r + (concreteData.theta_s - concreteData.theta_r) * S_e(h)

def specific_moisture_capacity(h):
    """
    Specific Moisture capacity / C
    """
    return (concreteData.theta_s - concreteData.theta_r) * concreteData.soil_water_retention * concreteData.n * concreteData.m * (concreteData.soil_water_retention * torch.abs(h))**(concreteData.n - 1) / S_e(h)

def hydraulic_conductivity(h):
    Se = S_e(h)
    return concreteData.K_s* Se**0.5* (1-(1-Se**(1/concreteData.m))**concreteData.m)**2

def residual_1d_head(x, y, scale: Scale):
    rescaled_h = y[:, 0] * scale.H 
    C = specific_moisture_capacity(rescaled_h) # [1/H]
    K = hydraulic_conductivity(rescaled_h)   # [L/T]
    #print('TRUE C', C.min().item(), C.max().item())
    #print('TRUE K', K.min().item(), K.max().item())

    C = C / (1 / scale.H)  
    K = K / (scale.L / scale.T)

    #print('rescaled_C', C.min().item(), C.max().item())
    #print('rescaled_K', K.min().item(), K.max().item())


    dh_dt = dde.grad.jacobian(y,x,i=0,j=1) * (scale.H/ scale.T)    # [H/T]
    dh_dz = dde.grad.jacobian(y,x,i=0,j=0) * (scale.H / scale.L)   # [H/L]
    time_term = C * dh_dt   # [1/H] * [H/T] = [1/T]


    Kdh_dz = K * dh_dz   # [L/T] * [H/L] = = [H/T]
    
    spatial_term = dde.grad.jacobian(Kdh_dz, x, i=0, j=0) * (scale.H/(scale.T*scale.L))  #[H/T]*[1/L] = [H/(LT)]  =ca [1/T]
    return time_term - spatial_term

def hydraulic_conductivity_theta(theta):
    Se = (theta - concreteData.theta_r) / (concreteData.theta_s - concreteData.theta_r)
    return concreteData.K_s* Se**0.5* (1-(1-Se**(1/concreteData.m))**concreteData.m)**2

def physical_possible_theta(theta):
    """
    Important because Conductivity makes weird shit if not PHYSICS working
    """
    return torch.clamp(theta, min=concreteData.theta_r + 1e-6, 
                        max=concreteData.theta_s - 1e-6)

def residual_1d_saturation(x, y, scale: Scale):
    theta = y[:, 0] # [theta]
    theta_physic = theta * scale.theta
    theta_phys_poss = physical_possible_theta(theta_physic)
    D = hydraulic_conductivity_theta(theta_phys_poss) / (scale.L / scale.T) #  [L/T]
    
    
    theta_t = dde.grad.jacobian(y, x, i=0, j=1) * (scale.theta / scale.T) #  [theta/T]
    dtheta_dz = dde.grad.jacobian(y, x, i=0, j=0) * (scale.theta / scale.L)# [theta/L]#
    Ddtheta_dz = D * dtheta_dz # [L/T] * [theta/L]  = [theta/T]
    spatial_term = dde.grad.jacobian(Ddtheta_dz, x, i=0, j=0) * (scale.theta / (scale.L*scale.T)) # [theta/T] * [1/L] = [theta/(TL)]#
    
    #print('----')
    #print('theta_scaled', theta.min().item(), theta.max().item())
    #print('theta_physic', theta_physic.min().item(), theta_physic.max().item())
    #print('theta_phys_poss', theta_phys_poss.min().item(), theta_phys_poss.max().item())
    #print('D', D.min().item(), D.max().item())
    #print('theta_t', theta_t.min().item(), theta_t.max().item())
    #print('spatial_term', spatial_term.min().item(), spatial_term.max().item())
    return theta_t - spatial_term


def residual_1d_mixed(x, y, scale: Scale):
    head = y[:,0:1]
    sat = y[:,1:2]
    res_head = residual_1d_head(x, head, scale)
    res_saturation = residual_1d_saturation(x, sat, scale)
    return [res_head, res_saturation]






# =========================================================================
# DARCY
# =========================================================================

def residual_2d_darcy(x, y):
    """
    Darcy's law for 2D flow
    """
    K = concreteData.K
    d2h_dx2 = dde.grad.hessian(y, x, i=0, j=0)  # dh/dx
    d2h_dy2 = dde.grad.hessian(y, x, i=0, j=1)  # dh/dy
    return -K * (d2h_dx2 + d2h_dy2)  # -K * grad(h)