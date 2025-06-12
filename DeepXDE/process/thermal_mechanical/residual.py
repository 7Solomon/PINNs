import torch
import deepxde as dde
from config import bernoulliBalken2DConfig, concreteData
from process.thermal_mechanical.scale import Scale
def residual_thermal_2d(x, y, scale: Scale):
    alpha_dT = concreteData.thermal_expansion_coefficient * y[:,2:3] * scale.Temperature  # [-]
    
    du_dx = dde.grad.jacobian(y, x, i=0, j=0)  # [-]
    dv_dy = dde.grad.jacobian(y, x, i=1, j=1)  # [-]
    du_dy = dde.grad.jacobian(y, x, i=0, j=1)  # [-]
    dv_dx = dde.grad.jacobian(y, x, i=1, j=0)  # [-]

    ex = du_dx + alpha_dT       # [-]
    ey = dv_dy + alpha_dT       # [-]
    exy = du_dy + dv_dx         # [-]
    
    strain_voigt = torch.cat([ex, ey, exy], dim=1)
    
    C_matrix = concreteData.C() / scale.sigma(concreteData.E)
    sigma_voigt = torch.matmul(strain_voigt, C_matrix.T)

    sigmax_x = dde.grad.jacobian(sigma_voigt, x, i=0, j=0)
    sigmay_y = dde.grad.jacobian(sigma_voigt, x, i=1, j=1)
    tauxy_y = dde.grad.jacobian(sigma_voigt, x, i=2, j=1)  
    tauxy_x = dde.grad.jacobian(sigma_voigt, x, i=2, j=0)  
    
 
    term_x = (sigmax_x + tauxy_y) 
    term_y = (sigmay_y + tauxy_x)
    
    T_t = dde.grad.jacobian(y, x, i=2, j=2)
    T_xx = dde.grad.hessian(y, x, i=2, j=0)
    T_yy = dde.grad.hessian(y, x, i=2, j=1)


    alpha_nd = concreteData.alpha() / (scale.L**2/scale.t)
    heat = T_t - alpha_nd * (T_xx + T_yy)
    #heat_physical = T_t - concreteData.alpha() * (T_xx + T_yy)  # [K/s]
    #heat = heat_physical / (scale.Temperature / scale.t)        # dimensionless
    
    #print('T_t:', T_t.min().item(), T_t.max().item())
    #print('T_xx:', T_xx.min().item(), T_xx.max().item())
    #print('T_yy:', T_yy.min().item(), T_yy.max().item())
    #print('term_x:', term_x.min().item(), term_x.max().item())
    #print('term_y:', term_y.min().item(), term_y.max().item())
    #print('heat:', heat.min().item(), heat.max().item())
    return [term_x, term_y, heat]