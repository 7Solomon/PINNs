import torch
import deepxde as dde
from config import bernoulliBalken2DConfig
from process.thermal_mechanical.scale import Scale

from material import concreteData
materialData = concreteData


def residual_thermal_2d(x, y, scale: Scale):
    #alpha_dT_nd = materialData.thermal_expansion_coefficient * y[:,2:3] / (1/scale.Temperature)  # [-]
    
    du_dx_nd = dde.grad.jacobian(y, x, i=0, j=0)  # [-]
    dv_dy_nd = dde.grad.jacobian(y, x, i=1, j=1)  # [-]
    du_dy_nd = dde.grad.jacobian(y, x, i=0, j=1)  # [-]
    dv_dx_nd = dde.grad.jacobian(y, x, i=1, j=0)  # [-]

    temp = scale.Temperature * y[:, 2:3]
    epsilon_th_nd = (1 + materialData.nu) * materialData.thermal_expansion_coefficient * temp / (scale.U/( scale.L))

    ex_mech_nd = du_dx_nd - epsilon_th_nd
    ey_mech_nd = dv_dy_nd - epsilon_th_nd
    exy_mech_nd = du_dy_nd + dv_dx_nd  # no shear in therm

    strain_voigt_nd = torch.cat([ex_mech_nd, ey_mech_nd, exy_mech_nd], dim=1)
    strain_voigt = strain_voigt_nd * (scale.U / scale.L)  # [-]
    
    C_matrix = materialData.C_stiffness_matrix()
    sigma_voigt = torch.matmul(strain_voigt, C_matrix.T)
    sigma_voigt_nd = sigma_voigt / scale.sigma  # [-]

    sigmax_x_nd = dde.grad.jacobian(sigma_voigt_nd, x, i=0, j=0)
    sigmay_y_nd = dde.grad.jacobian(sigma_voigt_nd, x, i=1, j=1)
    tauxy_y_nd = dde.grad.jacobian(sigma_voigt_nd, x, i=2, j=1)  
    tauxy_x_nd = dde.grad.jacobian(sigma_voigt_nd, x, i=2, j=0)  
    
 
    term_x = (sigmax_x_nd + tauxy_y_nd) 
    term_y = (sigmay_y_nd + tauxy_x_nd)

    # Heat
    T_t_nd = dde.grad.jacobian(y, x, i=2, j=2)
    T_xx_nd = dde.grad.hessian(y, x, i=2, j=0)
    T_yy_nd = dde.grad.hessian(y, x, i=2, j=1)

    pi_two = (materialData.alpha_thermal_diffusivity * scale.t) / (scale.L**2)

    heat = T_t_nd - (T_xx_nd + T_yy_nd) * pi_two

    return [term_x, term_y, heat]