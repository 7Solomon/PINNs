from process.moisture.scale import Scale
from config import richards1DConfig
import torch
import deepxde as dde


def S_e(h):
    """
    Effective saturation (VG)
    """
    core = (1 + (richards1DConfig.alpha * torch.abs(h))**richards1DConfig.n)**(-richards1DConfig.m)
    
    smooth = torch.sigmoid(-h)  # ca 1 for h < 0, ca 0 for h > 0
    S_e = smooth * core + (1 - smooth)  # diffrentiable switch
    return torch.clamp(S_e, min=1e-7, max=1.0 - 1e-7) 


def water_retention_curve(h):
    """
    Computes the volumetric water content (VWC) out of head.
    """
    return richards1DConfig.theta_r + (richards1DConfig.theta_s - richards1DConfig.theta_r) * S_e(h)

def specific_moisture_capacity(h):
    """
    Specific Moisture capacity / C
    """
    return (richards1DConfig.theta_s - richards1DConfig.theta_r) * richards1DConfig.alpha * richards1DConfig.n * richards1DConfig.m * (richards1DConfig.alpha * torch.abs(h))**(richards1DConfig.n - 1) / S_e(h)

def hydraulic_conductivity(h):
    Se = S_e(h)
    return richards1DConfig.K_s* Se**0.5* (1-(1-Se**(1/richards1DConfig.m))**richards1DConfig.m)**2

def residual_1d_head(x, y, scale: Scale):
    rescaled_h = y[:, 0] * scale.H 
    C = specific_moisture_capacity(rescaled_h)
    K = hydraulic_conductivity(rescaled_h)


    dh_dt = dde.grad.jacobian(y,x,i=0,j=1) / (scale.H/ scale.T)
    dh_dz = dde.grad.jacobian(y,x,i=0,j=0) / (scale.H / scale.L)
    time_term = C * dh_dt

    Kdh_dz = K * dh_dz
    spatial_term = dde.grad.jacobian(Kdh_dz, x, i=0, j=0) / (scale.H / scale.L)
    return time_term - spatial_term

def hydraulic_conductivity_theta(theta):
    Se = (theta - richards1DConfig.theta_r) / (richards1DConfig.theta_s - richards1DConfig.theta_r)
    return richards1DConfig.K_s* Se**0.5* (1-(1-Se**(1/richards1DConfig.m))**richards1DConfig.m)**2

def residual_1d_saturation(x, y, scale: Scale):
    # Extract theta (assuming y[:, 0] is the moisture content)
    theta = y[:, 0]
    
    # Time derivative (properly scaled)
    theta_t = dde.grad.jacobian(y, x, i=0, j=1) / scale.T
    
    # Hydraulic conductivity (check for valid theta values)
    theta_clamped = torch.clamp(theta, min=richards1DConfig.theta_r + 1e-6, 
                                max=richards1DConfig.theta_s - 1e-6)
    D = hydraulic_conductivity_theta(theta_clamped)
    
    # Spatial derivatives (properly scaled)
    dtheta_dz = dde.grad.jacobian(y, x, i=0, j=0) / scale.L
    Ddtheta_dz = D * dtheta_dz
    
    spatial_term = dde.grad.jacobian(Ddtheta_dz, x, i=0, j=0) / scale.L
    return theta_t - spatial_term