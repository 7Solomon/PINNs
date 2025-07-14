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
    h = h_nd * scale.H
    
    C = specific_moisture_capacity(h)  # [1/m]
    K= hydraulic_conductivity_head(h)  # [m/s]
    
    C_nd = C * scale.H  # [-]
    K_nd = K / materialData.K_s  # [-]
    
    dh_dt_nd = dde.grad.jacobian(y, x, i=0, j=1)  # [-]
    dh_dz_nd = dde.grad.jacobian(y, x, i=0, j=0)  # [-]
    
    # Dimensionless params
    pi_one = (materialData.K_s * scale.T * scale.H) / (scale.L**2)
    gravity_term = scale.L / scale.H  # [-]

    # Dimensionless flux
    flux_nd = K_nd * (dh_dz_nd + gravity_term)  # [-]

    spatial_term = pi_one * dde.grad.jacobian(flux_nd, x, i=0, j=0)  # [-]
    time_term = C_nd * dh_dt_nd  # [-]

    #print('--------')
    #print('h_nd', h_nd.min().item(), h_nd.max().item())
    #print('h', h.min().item(), h.max().item())
    #print('C_nd', C_nd.min().item(), C_nd.max().item())
    #print('K_nd', K_nd.min().item(), K_nd.max().item())
    #print('dh_dt_nd', dh_dt_nd.min().item(), dh_dt_nd.max().item())
    #print('dh_dz_nd', dh_dz_nd.min().item(), dh_dz_nd.max().item())
    #print('flux_nd', flux_nd.min().item(), flux_nd.max().item())
    #print('spatial_term', spatial_term.min().item(), spatial_term.max().item())
    #print('time_term', time_term.min().item(), time_term.max().item())
    
    return time_term - spatial_term

#def volumetric_water_content_saturation(theta):
#    Se = (theta - materialData.theta_r) / (materialData.theta_s - materialData.theta_r)
#    return materialData.K_s* Se**0.5* (torch.clamp(1-(1-Se**(1/materialData.m_vg)), min=1e-6)**materialData.m_vg)**2
#
#def physical_possible_S(S):
#    """
#    Important because Conductivity makes weird shit if not PHYSICS working
#    """
#    return torch.clamp(S, min=1e-6, max=1- 1e-6)
#
#def efective_saturation_from_saturation(S):
#    return (S - materialData.Sr) / (1 - materialData.Sr)
#
#def head(Se):
#    core = (Se**(-(1/materialData.m_vg))-1)**(1/materialData.n_vg) * (1/materialData.alpha_vg) 
#    smooth = torch.sigmoid(-20 * (Se - 1.0))
#    h = smooth * core
#    return h
#
#def hydraulic_conductivity_saturation(Se):
#    return Se**(1/2)*(1-(1-Se**(1/materialData.m_vg))**materialData.m_vg)**2

def get_theta_from_Se(Se):
    """Calculates volumetric water content (theta) from effective saturation (Se)."""
    return materialData.theta_r + (materialData.theta_s - materialData.theta_r) * Se

def get_head_from_Se(Se):
    """Calculates pressure head (h) from effective saturation (Se)."""
    Se_clamped = torch.clamp(Se, min=1e-7, max=1.0 - 1e-7)    
    term = (Se_clamped**(-1/materialData.m_vg)) - 1
    term_clamped = torch.clamp(term, min=0.0)
    h = -(term_clamped**(1/materialData.n_vg)) / materialData.alpha_vg
    return h

def get_K_from_Se(Se):
    """Calculates hydraulic conductivity (K) from effective saturation (Se)."""
    Se_clamped = torch.clamp(Se, min=1e-7, max=1.0 - 1e-7)
    term = 1 - (1 - Se_clamped**(1/materialData.m_vg))**materialData.m_vg
    K_r = Se_clamped**0.5 * torch.clamp(term, min=0.0)**2    
    return materialData.K_s * K_r

def residual_1d_saturation(x, y, scale: SaturationScale):
    Se_nd = y[:, 0:1]

    h_phys = get_head_from_Se(Se_nd)
    K_phys = get_K_from_Se(Se_nd)

    h_nd = h_phys / scale.H
    K_nd = K_phys / materialData.K_s

    gravity_term = scale.L / scale.H
    dh_dz_nd = dde.grad.jacobian(h_nd, x, i=0, j=0)
    flux_nd = K_nd * (dh_dz_nd + gravity_term)  # [-]

    spatial_term = dde.grad.jacobian(flux_nd, x, i=0, j=0)

    pi_group = (materialData.K_s * scale.T * scale.H) / (scale.L**2 * scale.theta)
    time_term = dde.grad.jacobian(Se_nd, x, i=0, j=1)

    #print('--------')
    #print('Se_nd', Se_nd.min().item(), Se_nd.max().item())
    #print('h_phys', h_phys.min().item(), h_phys.max().item())
    #print('K_phys', K_phys.min().item(), K_phys.max().item())
    #print('h_nd', h_nd.min().item(), h_nd.max().item())
    #print('K_nd', K_nd.min().item(), K_nd.max().item())
    #print('gravity_term', gravity_term)
    #print('dh_dz_nd', dh_dz_nd.min().item(), dh_dz_nd.max().item())
    #print('flux_nd', flux_nd.min().item(), flux_nd.max().item())
    #print('spatial_term', spatial_term.min().item(), spatial_term.max().item())
    #print('pi_group', pi_group)
    #print('time_term', time_term.min().item(), time_term.max().item())

    return time_term - pi_group * spatial_term





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