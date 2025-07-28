import torch
import deepxde as dde
from process.thermal_mechanical.scale import Scale

from material import concreteData
materialData = concreteData


def residual_thermal_2d(x, y, scale: Scale):
    temp_nd = y[:, 2:3] 
    temp_physical = temp_nd * scale.Temperature
    
    epsilon_th = materialData.thermal_expansion_coefficient * temp_physical
    epsilon_th_nd = epsilon_th / (scale.U / scale.L)
    
    du_dx_nd = dde.grad.jacobian(y, x, i=0, j=0)  # [-]
    dv_dy_nd = dde.grad.jacobian(y, x, i=1, j=1)  # [-]
    du_dy_nd = dde.grad.jacobian(y, x, i=0, j=1)  # [-]
    dv_dx_nd = dde.grad.jacobian(y, x, i=1, j=0)  # [-]


    ex_mech_nd = du_dx_nd - epsilon_th_nd
    ey_mech_nd = dv_dy_nd - epsilon_th_nd
    exy_mech_nd = du_dy_nd + dv_dx_nd  # No thermal shear strain

    strain_voigt_nd = torch.cat([ex_mech_nd, ey_mech_nd, exy_mech_nd], dim=1)
    strain_voigt = strain_voigt_nd * (scale.U / scale.L)  # [-]
    
    C_matrix = materialData.C_stiffness_matrix()
    sigma_voigt = torch.matmul(strain_voigt, C_matrix.T)
    sigma_voigt_nd = sigma_voigt / scale.sigma  # [-]

    sigmax_x_nd = dde.grad.jacobian(sigma_voigt_nd, x, i=0, j=0)
    sigmay_y_nd = dde.grad.jacobian(sigma_voigt_nd, x, i=1, j=1)
    tauxy_y_nd = dde.grad.jacobian(sigma_voigt_nd, x, i=2, j=1)  
    tauxy_x_nd = dde.grad.jacobian(sigma_voigt_nd, x, i=2, j=0)  
    
    #body_force = (scale.L / scale.sigma) * (-materialData.rho * materialData.g)  # [m/s^2]

    term_x = (sigmax_x_nd + tauxy_y_nd) 
    term_y = (sigmay_y_nd + tauxy_x_nd)

    # Heat
    T_t_nd = dde.grad.jacobian(y, x, i=2, j=2)
    T_xx_nd = dde.grad.hessian(y, x, i=2, j=0)
    T_yy_nd = dde.grad.hessian(y, x, i=2, j=1)

    pi_two = (materialData.alpha_thermal_diffusivity * scale.t) / (scale.L**2)

    heat = T_t_nd - (T_xx_nd + T_yy_nd) * pi_two
    #print()
    #print('pi_two:', pi_two)
    #print('-----')
    #print('u', y[:, 0].min().item(), y[:, 0].max().item())
    #print('v', y[:, 1].min().item(), y[:,1].max().item())
    #print('sigmax_x:', sigmax_x_nd.min().item(), sigmax_x_nd.max().item())
    #print('sigmay_y:', sigmay_y_nd.min().item(), sigmay_y_nd.max().item())
    #print('tauxy_y:', tauxy_y_nd.min().item(), tauxy_y_nd.max().item())
    #print('tauxy_x:', tauxy_x_nd.min().item(), tauxy_x_nd.max().item())
    #print('b_force:', (scale.L / scale.sigma) * (-materialData.rho * materialData.g))
    #print('scale.sigma:', scale.sigma)
    #print('scale.L:', scale.L)
    #print('scale.U:', scale.U)
    #print('term_x:', term_x.min().item(), term_x.max().item())
    #print('term_y:', term_y.min().item(), term_y.max().item())
    #print('heat:', heat.min().item(), heat.max().item())
    return [term_x, term_y, heat]