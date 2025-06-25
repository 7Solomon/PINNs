from process.moisture.scale import *
#from config import richards1DConfig
import torch
import deepxde as dde

#from config import materialData
from material import concreteData, sandData
materialData = concreteData

def S_e(h):
    """
    Effective saturation (VG)
    """
    core = (1 + (materialData.alpha_vg * torch.abs(h))**materialData.n_vg)**(-materialData.m_vg)

    smooth = torch.sigmoid(-h)  # ca 1 for h < 0, ca 0 for h > 0
    S_e = smooth * core + (1 - smooth)  # diffrentiable switch
    return torch.clamp(S_e, min=1e-7, max=1.0 - 1e-7) 


def water_retention_curve(h):
    """
    Computes the volumetric water content (VWC) out of head.
    """
    return materialData.theta_r + (materialData.theta_s - materialData.theta_r) * S_e(h)

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
        
        term1 = (materialData.theta_s - materialData.theta_r)
        term2 = materialData.alpha_vg * materialData.n_vg * materialData.m_vg
        term3 = (materialData.alpha_vg * torch.abs(h_unsat))**(materialData.n_vg - 1)
        term4 = (1 + (materialData.alpha_vg * torch.abs(h_unsat))**materialData.n_vg)**(-(materialData.m_vg + 1))

        C[unsaturated_mask] = term1 * term2 * term3 * term4

    return C
def hydraulic_conductivity_head(h):
    Se = S_e(h)
    #print('Se', Se.min().item(), Se.max().item())
    return materialData.K_s* Se**0.5* (1-(1-Se**(1/materialData.m_vg))**materialData.m_vg)**2

def residual_1d_head(x, y, scale: HeadScale):
    h_nd = y[:, 0]  # [-]
    h = h_nd * scale.h_char
    
    C = specific_moisture_capacity(h)  # [1/m]
    K= hydraulic_conductivity_head(h)  # [m/s]
    
    C_nd = C * scale.h_char  # [-]
    K_nd = K / materialData.K_s  # [-]
    
    dh_dt_nd = dde.grad.jacobian(y, x, i=0, j=1)  # [-]
    dh_dz_nd = dde.grad.jacobian(y, x, i=0, j=0)  # [-]
    
    # Dimensionless params
    Da = scale.Da_hydraulic
    Pe = scale.Pe
    
    # Dimensionless flux
    flux_star = K_nd * (dh_dz_nd + Pe)  # [-]
    
    # Spatial term
    spatial_term = (Da/Pe) * dde.grad.jacobian(flux_star, x, i=0, j=0)  # [-]
    
    # Time term
    time_term = C_nd * dh_dt_nd  # [-]
    
    return time_term - spatial_term

def volumetric_water_content_saturation(theta):
    Se = (theta - materialData.theta_r) / (materialData.theta_s - materialData.theta_r)
    return materialData.K_s* Se**0.5* (torch.clamp(1-(1-Se**(1/materialData.m_vg)), min=1e-6)**materialData.m_vg)**2

def physical_possible_S(S):
    """
    Important because Conductivity makes weird shit if not PHYSICS working
    """
    return torch.clamp(S, min=1e-6, max=1- 1e-6)

def efective_saturation_theta(S):
    return (S - materialData.Sr()) / (1 - materialData.Sr())

def head(Se):
    core = (Se**(-(1/materialData.m_vg))-1)**(1/materialData.n_vg) * (1/materialData.alpha_vg) 
    smooth = torch.sigmoid(-20 * (Se - 1.0))
    h = smooth * core
    return h

def hydraulic_conductivity_saturation(Se):
    return Se**(1/2)*(1-(1-Se**(1/materialData.m_vg))**materialData.m_vg)**2


def residual_1d_saturation(x, y, scale: SaturationScale):
    S = y[:, 0:1] * scale.S # [S]
    S_phys_poss = physical_possible_S(S)
    efective_saturation = efective_saturation_theta(S_phys_poss)

    theta = volumetric_water_content_saturation(S_phys_poss) / scale.theta
    h = head(efective_saturation) / scale.H
    K = hydraulic_conductivity_saturation(efective_saturation) / scale.K 

    time_term = dde.grad.jacobian(theta, x, i=0, j=1) 
    
    dh_dz = dde.grad.jacobian(h, x, i=0, j=0)
    flux_term = K * dh_dz
    spatial_term = dde.grad.jacobian(flux_term, x, i=0, j=0)
    pi_one = (scale.H*scale.K*scale.T)/(scale.L**2 *scale.theta)

    print('----')
    print('S', S.min().item(), S.max().item())
    print('S_phys_poss', S_phys_poss.min().item(), S_phys_poss.max().item())
    print('efective_saturation', efective_saturation.min().item(), efective_saturation.max().item())
    print('theta', theta.min().item(), theta.max().item())
    print('h', h.min().item(), h.max().item())
    print('K', K.min().item(), K.max().item())
    print('time_term', time_term.min().item(), time_term.max().item())
    print('dh_dz', dh_dz.min().item(), dh_dz.max().item())
    print('flux_term', flux_term.min().item(), flux_term.max().item())
    print('spatial_term', spatial_term.min().item(), spatial_term.max().item())
    print('pi_one * spatial_term', (pi_one*spatial_term).min().item(), (spatial_term* pi_one).max().item())
    print('pi_one', pi_one)
    print('ScaleL', scale.L)
    print('ScaleT', scale.T)
    print('ScaleS', scale.S)
    print('ScaleH', scale.H)
    print('ScaleK', scale.K)
    print('ScaleTheta', scale.theta)
    return time_term - spatial_term * pi_one


# --- SUGGESTED REVISIONS ---

# Let's assume `materialData` contains these physical constants:
# materialData.theta_r: Residual water content
# materialData.theta_s: Saturated water content
# materialData.K_s: Saturated hydraulic conductivity [L/T]
# materialData.alpha: van Genuchten alpha parameter [1/L]
# materialData.n: van Genuchten n parameter
# materialData.m: 1 - 1/n

def get_effective_saturation(S_total):
    """Calculates effective saturation (Se) from total saturation (S)."""
    theta = get_volumetric_water_content(S_total)
    return (theta - materialData.theta_r) / (materialData.theta_s - materialData.theta_r)


def get_volumetric_water_content(S_total):
    """Calculates volumetric water content (theta) from total saturation (S)."""
    return S_total * (materialData.theta_s - materialData.theta_r) + materialData.theta_r

def get_pressure_head(Se):
    """Calculates pressure head (h) from effective saturation (Se)."""
    Se_clamped = torch.clamp(Se, min=1e-7, max=1.0 - 1e-7)
    core = ((Se_clamped**(-1/materialData.m_vg)) - 1)**(1/materialData.n_vg) / materialData.alpha_vg

    # Smoothly force h 0 as Se 1
    smooth = torch.sigmoid(-20 * (Se - 1.0))
    h = smooth * core
    return h

def get_hydraulic_conductivity(Se):
    """Calculates hydraulic conductivity (K) from effective saturation (Se)."""
    Se_clamped = torch.clamp(Se, min=1e-7, max=1.0 - 1e-7)
    K_r = Se_clamped**0.5 * (1 - (1 - Se_clamped**(1/materialData.m_vg))**materialData.m_vg)**2
    return materialData.K_s * K_r

def residual_1d_saturation(x, y, scale: SaturationScale):
    S_total = y[:, 0:1]
    S_clamped = physical_possible_S(S_total)

    Theta = get_effective_saturation(S_clamped)
    
    h_physical = get_pressure_head(Theta)
    K_physical = get_hydraulic_conductivity(Theta)

    H = h_physical / scale.H # [-] Head
    K = K_physical / scale.K # [-] Conductivity

    time_term = dde.grad.jacobian(Theta, x, i=0, j=1)
    dH_dZ = dde.grad.jacobian(H, x, i=0, j=0)

    flux_term = K * dH_dZ
    diffusion_term = dde.grad.jacobian(flux_term, x, i=0, j=0)

    pi_one = (scale.K * scale.H * scale.T) / (scale.L**2 * (materialData.theta_s - materialData.theta_r))
    spatial_term = pi_one * diffusion_term
    #print('----')
    #print('S_total', S_total.min().item(), S_total.max().item())
    #print('S_clamped', S_clamped.min().item(), S_clamped.max().item())
    #print('Theta', Theta.min().item(), Theta.max().item())
    #print('h_physical', h_physical.min().item(), h_physical.max().item())
    #print('K_physical', K_physical.min().item(), K_physical.max().item())
    #print('H', H.min().item(), H.max().item())
    #print('K', K.min().item(), K.max().item())
    #print('dH_dZ', dH_dZ.min().item(), dH_dZ.max().item())
    #print('flux_term', flux_term.min().item(), flux_term.max().item())
    #print('diffusion_term', diffusion_term.min().item(), diffusion_term.max().item())
    #print('pi_one', pi_one)
    #print('spatial_term', spatial_term.min().item(), spatial_term.max().item())
    #print('time_term', time_term.min().item(), time_term.max().item())
    #print('ScaleL', scale.L)
    #print('ScaleT', scale.T)
    #print('ScaleS', scale.S)
    #print('ScaleH', scale.H)
    #print('ScaleK', scale.K)
    #print('ScaleTheta', scale.theta)
    
    return  time_term - spatial_term
#def residual_1d_mixed(x, y, scale: Scale):
#    head = y[:,0:1]
#    sat = y[:,1:2]
#    res_head = residual_1d_head(x, head, scale)
#    res_saturation = residual_1d_saturation(x, sat, scale)
#    return [res_head, res_saturation]






# =========================================================================
# DARCY
# =========================================================================

def residual_2d_darcy(x, y):
    """
    Darcy's law for 2D flow
    """
    K = materialData.K_s # Hydraulic conductivity [m/s]
    d2h_dx2 = dde.grad.hessian(y, x, i=0, j=0)  # dh/dx
    d2h_dy2 = dde.grad.hessian(y, x, i=0, j=1)  # dh/dy
    return -K * (d2h_dx2 + d2h_dy2)  # -K * grad(h)