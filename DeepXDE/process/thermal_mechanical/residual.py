import torch
import deepxde as dde
from config import bernoulliBalken2DConfig, concreteData
from process.thermal_mechanical.scale import Scale

def residual_thermal_2d(x, y, scale: Scale): # x : [x,y,t] , y : [u,v,T]
    du_dx = dde.grad.jacobian(y, x, i=0, j=0)
    dv_dy = dde.grad.jacobian(y, x, i=1, j=1)
    du_dy = dde.grad.jacobian(y, x, i=0, j=1)
    dv_dx = dde.grad.jacobian(y, x, i=1, j=0)
    
    alpha_dT = concreteData.thermal_expansion_coefficient * y[:,2:3]
    
    ex = du_dx + alpha_dT
    ey = dv_dy + alpha_dT
    exy = du_dy + dv_dx 
    
    strain_voigt = torch.cat([ex, ey, exy], dim=1)
    
    C_matrix = bernoulliBalken2DConfig.C(concreteData)
    sigma_voigt = torch.matmul(strain_voigt, C_matrix.T) 

    scaled_sigma_char = scale.sigma_voigt(concreteData.E, concreteData.thermal_expansion_coefficient)
    scaled_sigma_voigt = sigma_voigt / scaled_sigma_char
    sigmax_x = dde.grad.jacobian(scaled_sigma_voigt, x, i=0, j=0) / (scaled_sigma_char[0]/ scale.Lx)
    sigmay_y = dde.grad.jacobian(scaled_sigma_voigt, x, i=1, j=1) / (scaled_sigma_char[1]/ scale.Ly)
    tauxy_y = dde.grad.jacobian(scaled_sigma_voigt, x, i=2, j=1) / (scaled_sigma_char[2]/ scale.Ly)
    tauxy_x = dde.grad.jacobian(scaled_sigma_voigt, x, i=2, j=0) / (scaled_sigma_char[2]/ scale.Lx)


    term_x = sigmax_x + tauxy_y
    term_y = sigmay_y + tauxy_x

    return [term_x, term_y]