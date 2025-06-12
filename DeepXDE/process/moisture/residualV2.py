from process.moisture.scale import Scale
#from config import richards1DConfig
import torch
import deepxde as dde

from config import concreteData
#from material import concreteData

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
    Specific Moisture capacity C(h) = d(theta)/dh for h < 0.
    For h >= 0, the derivative is 0.
    """
    # unsaturated zone (h < 0) mas
    unsaturated_mask = h < 0
    C = torch.zeros_like(h)
    
    if torch.any(unsaturated_mask):
        h_unsat = h[unsaturated_mask]
        
        term1 = (concreteData.theta_s - concreteData.theta_r)
        term2 = concreteData.soil_water_retention * concreteData.n * concreteData.m
        term3 = (concreteData.soil_water_retention * torch.abs(h_unsat))**(concreteData.n - 1)
        term4 = (1 + (concreteData.soil_water_retention * torch.abs(h_unsat))**concreteData.n)**(-(concreteData.m + 1))
        
        C[unsaturated_mask] = term1 * term2 * term3 * term4

    return C
def hydraulic_conductivity(h):
    Se = S_e(h)
    #print('Se', Se.min().item(), Se.max().item())
    return concreteData.K_s* Se**0.5* (1-(1-Se**(1/concreteData.m))**concreteData.m)**2

def residual_1d_head(x, y, scale: Scale):
    rescaled_h = y[:, 0] * scale.H 
    C = specific_moisture_capacity(rescaled_h)  # [1/m]
    K = hydraulic_conductivity(rescaled_h)      # [m/s]

    log_K = torch.log(K + 1e-20)          # smoothing
    K_smoothed = torch.exp(log_K)        # 

    # Convert dimensionless
    C_scaled = C * scale.H   # [1/m] * [m] = [-]
    K_scaled = K_smoothed / scale.K  # [m/s] / [m/s] = [-]


    dh_dt = dde.grad.jacobian(y,x,i=0,j=1)  # [-]
    dh_dz = dde.grad.jacobian(y,x,i=0,j=0)  # [-]
    
    time_term = C_scaled * dh_dt #* ((scale.H*scale.L)/(scale.K*scale.T)) # [now -]

    pi_one = (scale.K*scale.T)/scale.L**2  
    flux_term = pi_one * K_scaled * dh_dz 
    spatial_term = dde.grad.jacobian(flux_term, x, i=0, j=0)

    #print('----')
    #print('rescaled_h', rescaled_h.min().item(), rescaled_h.max().item())
    #print('C', C.min().item(), C.max().item())
    #print('K', K.min().item(), K.max().item())
    #print('C_scaled', C_scaled.min().item(), C_scaled.max().item())
    #print('K_scaled', K_scaled.min().item(), K_scaled.max().item())
    #print('dh_dt', dh_dt.min().item(), dh_dt.max().item())
    #print('dh_dz', dh_dz.min().item(), dh_dz.max().item())
    #print('time_term', time_term.min().item(), time_term.max().item())
    #print('spatial_term', spatial_term.min().item(), spatial_term.max().item())
    #print('ScaleL', scale.L)
    #print('ScaleT', scale.T)
    #print('ScaleH', scale.H)
    #print('ScaleK', scale.K)
    #print('pi_one', pi_one)
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