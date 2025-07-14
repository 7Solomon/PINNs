import math
from process.mechanic.scale import MechanicScale
from utils.functions import voigt_to_tensor
import deepxde as dde
import torch
from config import bernoulliBalkenTConfig, cooksMembranConfig, bernoulliBalken2DConfig
from material import concreteData
    
def pde_1d_residual(x, y):
  w_x = dde.grad.jacobian(y,x, i=0)
  w_xx = dde.grad.jacobian(w_x, x, i=0)
  w_xxx = dde.grad.jacobian(w_xx, x, i=0)
  w_xxxx = dde.grad.jacobian(w_xxx, x, i=0)
  return w_xxxx - 1.0

  #return bernoulliBalkenConfig.EI*w_xxxx - bernoulliBalkenConfig.f(x[:,0], x[:,1])
def pde_2d_residual(x, y, scale: MechanicScale):
  e_x_nd = dde.grad.jacobian(y,x, i=0, j=0)
  e_y_nd = dde.grad.jacobian(y,x, i=1, j=1)
  g_xy_nd = dde.grad.jacobian(y,x, i=0, j=1) + dde.grad.jacobian(y,x, i=1, j=0)
  e_voigt_nd = torch.cat([e_x_nd, e_y_nd, g_xy_nd], dim=1)

  e_voigt  = e_voigt_nd * (scale.U / scale.L)
  C = concreteData.C_stiffness_matrix()
  sigma_voigt_nd = torch.matmul(e_voigt, C) / scale.sigma  # [1/L**2]

  sigmax_x_nd = dde.grad.jacobian(sigma_voigt_nd, x, i=0, j=0)  
  sigmay_y_nd = dde.grad.jacobian(sigma_voigt_nd, x, i=1, j=1) 
  tauxy_y_nd = dde.grad.jacobian(sigma_voigt_nd, x, i=2, j=1)
  tauxy_x_nd = dde.grad.jacobian(sigma_voigt_nd, x, i=2, j=0) 

  b_force = (scale.L / scale.sigma) * (-concreteData.rho * concreteData.g)

  #print('-----')
  #print('u', y[:,0].min().item(), y[:,0].max().item())
  #print('v', y[:,1].min().item(), y[:,1].max().item())
  #print('sigmax_x:', sigmax_x_nd.min().item(), sigmax_x_nd.max().item())
  #print('sigmay_y:', sigmay_y_nd.min().item(), sigmay_y_nd.max().item())
  #print('tauxy_y:', tauxy_y_nd.min().item(), tauxy_y_nd.max().item())
  #print('tauxy_x:', tauxy_x_nd.min().item(), tauxy_x_nd.max().item())
  #print('b_force:', b_force)
  #print('scale.sigma:', scale.sigma)
  #print('scale.L:', scale.L)
  #print('scale.U:', scale.U)
  #print()

  return [sigmax_x_nd + tauxy_y_nd, sigmay_y_nd + tauxy_x_nd + b_force] #- 1.0/scale.f]



def pde_1d_t_residual(x, y): 
  w_tt = dde.grad.hessian(y,x, i=1,j=1)
  w_xx = dde.grad.hessian(y,x, i=0,j=0)
  w_xxxx = dde.grad.hessian(w_xx,x, i=0,j=0)
  return w_tt + w_xxxx - bernoulliBalkenTConfig.f(x[:,0], x[:,1])
  #return 10**2*w_tt + 4 * 10**6*w_xxxx - 4 * 10**6*(1-16*math.pi**2)*math.sin(x[0]/math.pi)*math.cos(800*x[1]/math.pi)/math.pi**3
  #return w_tt - bernoulliBalkenTConfig.EI*w_xxxx - bernoulliBalkenTConfig.f(x[:,0], x[:,1])
def calc_sigma(x,y):
  pass

#def cooks_residual(x,y):
#
#  du_dx = dde.grad.jacobian(y, x, i=0, j=0)
#  dv_dy = dde.grad.jacobian(y, x, i=1, j=1)
#  du_dy = dde.grad.jacobian(y, x, i=0, j=1)
#  dv_dx = dde.grad.jacobian(y, x, i=1, j=0)
#
#  e_voigt = torch.cat([du_dx, dv_dy, du_dy + dv_dx], dim=1)
#  sigma_voigt = torch.matmul(e_voigt, cooksMembranConfig.C())
#
#  sigma = voigt_to_tensor(sigma_voigt)
#  #sigma = sigma 
#
#  dsigma_dx = dde.grad.jacobian(sigma, x, i=0)
#  dsigma_dy = dde.grad.jacobian(sigma, x, i=1)
#
#  return dsigma_dx + dsigma_dy # + cooksMembranConfig.f(x)

def pde_2d_ensemble_residual(x, y, scale: MechanicScale):
    e_x_nd = dde.grad.jacobian(y, x, i=0, j=0)
    e_y_nd = dde.grad.jacobian(y, x, i=1, j=1)
    g_xy_nd = dde.grad.jacobian(y, x, i=0, j=1) + dde.grad.jacobian(y, x, i=1, j=0)
    e_voigt_nd = torch.cat([e_x_nd, e_y_nd, g_xy_nd], dim=1)

    e_voigt = e_voigt_nd * (scale.U / scale.L)
    C = concreteData.C_stiffness_matrix()
    sigma_voigt_nd = torch.matmul(e_voigt, C) / scale.sigma

    # Stress divergence from constitutive law
    sigmax_x_nd = dde.grad.jacobian(sigma_voigt_nd, x, i=0, j=0)
    sigmay_y_nd = dde.grad.jacobian(sigma_voigt_nd, x, i=1, j=1)
    tauxy_y_nd = dde.grad.jacobian(sigma_voigt_nd, x, i=2, j=1)
    tauxy_x_nd = dde.grad.jacobian(sigma_voigt_nd, x, i=2, j=0)

    # Stress divergence from predicted stress fields
    sigmax_x_nd_pred = dde.grad.jacobian(y, x, i=2, j=0)
    sigmay_y_nd_pred = dde.grad.jacobian(y, x, i=3, j=1)
    tauxy_y_nd_pred = dde.grad.jacobian(y, x, i=4, j=1)
    tauxy_x_nd_pred = dde.grad.jacobian(y, x, i=4, j=0)

    # Equilibrium equations 
    equilibrium_x_constitutive = sigmax_x_nd + tauxy_y_nd
    equilibrium_y_constitutive = sigmay_y_nd + tauxy_x_nd  # No body  # - matherialData.rho * materialData.g
    
    equilibrium_x_predicted = sigmax_x_nd_pred + tauxy_y_nd_pred
    equilibrium_y_predicted = sigmay_y_nd_pred + tauxy_x_nd_pred

    # Consistency equations
    consistency_sigma_x = sigma_voigt_nd[:, 0:1] - y[:, 2:3]
    consistency_sigma_y = sigma_voigt_nd[:, 1:2] - y[:, 3:4]
    consistency_tau_xy = sigma_voigt_nd[:, 2:3] - y[:, 4:5]

    return [
        equilibrium_x_constitutive,
        equilibrium_x_predicted,
        equilibrium_y_constitutive,
        equilibrium_y_predicted,
        consistency_sigma_x,
        consistency_sigma_y,
        consistency_tau_xy
    ]
