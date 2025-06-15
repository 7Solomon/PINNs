
import deepxde as dde
import torch
from process.mechanical_moisture.scale import Scale

from material import concreteData

materialData = concreteData

def soil_water_retention_curve(theta):
    s_eff = (theta - materialData.theta_r) / (materialData.theta_s - materialData.theta_r)
    s_eff = torch.clamp(s_eff, min=1e-6, max=1.0 - 1e-6)
    m = materialData.m_vg
    n = materialData.n_vg
    
    head = (1.0 / materialData.alpha_vg) * (s_eff**(-1.0/m) - 1.0)**(1.0/n)
    return -head
def pressure(theta):
    return materialData.rho_w*materialData.g*soil_water_retention_curve(theta) # [pa]

def moisture_diffusivity(theta, e):
    """
    This is change of diffusivity with strain, and not porosity.
    """
    return materialData.D_moisture * (1 + materialData.strain_moisture_coulling_coef * (e[:,0:1] + e[:,1:2]))

def residual(x, y, scale: Scale):
    e_x = dde.grad.jacobian(y,x, i=0, j=0)
    e_y = dde.grad.jacobian(y,x, i=1, j=1)
    g_xy = dde.grad.jacobian(y,x, i=0, j=1) + dde.grad.jacobian(y,x, i=1, j=0)
    e_voigt = torch.cat([e_x, e_y, g_xy], dim=1)
    C_scaled = materialData.C_stiffness_matrix() / scale.sigma # C: [N/L**2]
    sigma_voigt = torch.matmul(e_voigt, C_scaled)   # [-]

    # stress with moisture
    phys_theta = y[:,2:3] * scale.theta
    moisture_pressure = (materialData.alpha_biot * pressure(phys_theta)) / scale.sigma  # [-]
    sigma_voigt_with_moisture = sigma_voigt - moisture_pressure

    sigmax_x = dde.grad.jacobian(sigma_voigt_with_moisture, x, i=0, j=0)  
    sigmay_y = dde.grad.jacobian(sigma_voigt_with_moisture, x, i=1, j=1) 
    tauxy_y = dde.grad.jacobian(sigma_voigt_with_moisture, x, i=2, j=1)
    tauxy_x = dde.grad.jacobian(sigma_voigt_with_moisture, x, i=2, j=0) 

    res_x = sigmax_x + tauxy_y
    res_y = sigmay_y + tauxy_x - 1.0/scale.f 
    
    vx = dde.grad.jacobian(y, x, i=0, j=2)
    vy = dde.grad.jacobian(y, x, i=1, j=2)

    dvx_dx = dde.grad.jacobian(vx, x, i=0, j=0)
    dvy_dy = dde.grad.jacobian(vy, x, i=0, j=1)
    e_voll = dvx_dx + dvy_dy

    coupling_coef = materialData.alpha_biot / (scale.theta / scale.epsilon)
    transient_coupling = coupling_coef * e_voll 

    # moisture
    phys_e_voigt = e_voigt * scale.epsilon 
    D = moisture_diffusivity(phys_theta, phys_e_voigt) / (scale.L**2/scale.t)
    theta_t = dde.grad.jacobian(y, x, i=2, j=2) 
    theta_x = dde.grad.jacobian(y, x, i=2, j=0)     
    theta_y = dde.grad.jacobian(y, x, i=2, j=1)

    D_theta_x = D * theta_x
    D_theta_y = D * theta_y

    div_x = dde.grad.jacobian(D_theta_x, x, i=0, j=0)
    div_y = dde.grad.jacobian(D_theta_y, x, i=0, j=1)
    res_theta = theta_t + transient_coupling - (div_x + div_y)

    #print('----')
    #print(f'sigma_voigt: {sigma_voigt.min().item()}, {sigma_voigt.max().item()}')
    #print(f'sigma_voigt_with_moisture: {sigma_voigt_with_moisture.min().item()}, {sigma_voigt_with_moisture.max().item()}')
    #print(f'sigmax_x: {sigmax_x.min().item()}, {sigmax_x.max().item()}')
    #print(f'sigmay_y: {sigmay_y.min().item()}, {sigmay_y.max().item()}')
    #print(f'tauxy_y: {tauxy_y.min().item()}, {tauxy_y.max().item()}')
    #print(f'tauxy_x: {tauxy_x.min().item()}, {tauxy_x.max().item()}')
    #print(f'res_x: {res_x.min().item()}, {res_x.max().item()}')
    #print(f'res_y: {res_y.min().item()}, {res_y.max().item()}')
    #print(f'vx: {vx.min().item()}, {vx.max().item()}')
    #print(f'vy: {vy.min().item()}, {vy.max().item()}')
    #print(f'dvx_dx: {dvx_dx.min().item()}, {dvx_dx.max().item()}')
    #print(f'dvy_dy: {dvy_dy.min().item()}, {dvy_dy.max().item()}')
    #print(f'e_voll: {e_voll.min().item()}, {e_voll.max().item()}')
    #print(f'coupling_coef: {coupling_coef}')
    #print(f'transient_coupling: {transient_coupling.min().item()}, {transient_coupling.max().item()}')
    #print(f'phys_theta: {phys_theta.min().item()}, {phys_theta.max().item()}')
    #print(f'D: {D.min().item()}, {D.max().item()}')
    #print(f'theta_t: {theta_t.min().item()}, {theta_t.max().item()}')
    #print(f'theta_x: {theta_x.min().item()}, {theta_x.max().item()}')
    #print(f'theta_y: {theta_y.min().item()}, {theta_y.max().item()}')
    #print(f'D_theta_x: {D_theta_x.min().item()}, {D_theta_x.max().item()}')
    #print(f'D_theta_y: {D_theta_y.min().item()}, {D_theta_y.max().item()}')
    #print(f'div_x: {div_x.min().item()}, {div_x.max().item()}')
    #print(f'div_y: {div_y.min().item()}, {div_y.max().item()}')
    #print(f'res_theta: {res_theta.min().item()}, {res_theta.max().item()}')
    #print(f'scale.L: {scale.L}, scale.t: {scale.t}, scale.epsilon: {scale.epsilon}, scale.theta: {scale.theta}')
    #print(f'scale.sigma: {scale.sigma}, scale.f: {scale.f}')

    return [res_x, res_y, res_theta]