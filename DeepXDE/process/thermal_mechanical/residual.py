import torch
import deepxde as dde
from config import bernoulliBalken2DConfig, concreteData
from process.thermal_mechanical.scale import Scale

def residual_thermal_2d(x, y, scale: Scale): # x : [x,y,t] , y : [u,v,T]

    du_dx = dde.grad.jacobian(y, x, i=0, j=0) / (1/scale.L) #* (scale.U(concreteData.thermal_expansion_coefficient))
    dv_dy = dde.grad.jacobian(y, x, i=1, j=1) / (1/scale.L) #* (scale.U(concreteData.thermal_expansion_coefficient))
    du_dy = dde.grad.jacobian(y, x, i=0, j=1) / (1/scale.L) #* (scale.U(concreteData.thermal_expansion_coefficient))
    dv_dx = dde.grad.jacobian(y, x, i=1, j=0) / (1/scale.L) #* (scale.U(concreteData.thermal_expansion_coefficient))

    scaled_alpha = concreteData.thermal_expansion_coefficient / (1/scale.Temperature)
    alpha_dT = scaled_alpha * y[:,2:3]

    #print('Temperature range (scaled):', y[:,2:3].min().item(), y[:,2:3].max().item())
    #print('scaled_alpha:', scaled_alpha)
    #print('alpha_dT range:', alpha_dT.min().item(), alpha_dT.max().item())

    
    ex = du_dx + alpha_dT
    ey = dv_dy + alpha_dT
    exy = du_dy + dv_dx 
    
    strain_voigt = torch.cat([ex, ey, exy], dim=1)
    
    C_matrix = bernoulliBalken2DConfig.C(concreteData)
    sigma_voigt = torch.matmul(strain_voigt, C_matrix.T) 
    #print('sigma_voigt0', sigma_voigt[:,0].min().item(), sigma_voigt[:,0].max().item())
    #print('sigma_voigt1', sigma_voigt[:,1].min().item(), sigma_voigt[:,1].max().item())
    #print('sigma_voigt2', sigma_voigt[:,2].min().item(), sigma_voigt[:,2].max().item())
    sigmax_x = dde.grad.jacobian(sigma_voigt, x, i=0, j=0) * (1/ scale.L**2)
    sigmay_y = dde.grad.jacobian(sigma_voigt, x, i=1, j=1) * (1/ scale.L**2)
    tauxy_y = dde.grad.jacobian(sigma_voigt, x, i=2, j=1) * (1/ scale.L**2)
    tauxy_x = dde.grad.jacobian(sigma_voigt, x, i=2, j=0) * (1/ scale.L**2)

    #print('sigmax_x', sigmax_x.min().item(), sigmax_x.max().item())
    #print('sigmay_y', sigmay_y.min().item(), sigmay_y.max().item())
    #print('tauxy_y', tauxy_y.min().item(), tauxy_y.max().item())
    #print('tauxy_x', tauxy_x.min().item(), tauxy_x.max().item())

    term_x = sigmax_x + tauxy_y
    term_y = sigmay_y + tauxy_x
    

    # just added not run through
    T_t=dde.grad.jacobian(y, x, i=2, j=2)
    T_xx=dde.grad.hessian(y, x, i=2, j=0)
    T_yy=dde.grad.hessian(y, x, i=2, j=1)   
    scaled_heat = concreteData.alpha() * (1 / scale.Temperature)
    heat = scaled_heat * T_t * (T_xx + T_yy)

    return [term_x, term_y, heat]