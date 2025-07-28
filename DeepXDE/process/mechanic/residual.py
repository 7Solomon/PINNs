import math
from process.mechanic.scale import EnsemnbleMechanicScale, MechanicScale
from utils.functions import voigt_to_tensor
import deepxde as dde
import torch
from config import bernoulliBalkenTConfig, cooksMembranConfig, bernoulliBalken2DConfig
from material import concreteData

materialData = concreteData
    
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
  C = materialData.C_stiffness_matrix()
  sigma_voigt_nd = torch.matmul(e_voigt, C) / scale.sigma  # [1/L**2]

  sigmax_x_nd = dde.grad.jacobian(sigma_voigt_nd, x, i=0, j=0)  
  sigmay_y_nd = dde.grad.jacobian(sigma_voigt_nd, x, i=1, j=1) 
  tauxy_y_nd = dde.grad.jacobian(sigma_voigt_nd, x, i=2, j=1)
  tauxy_x_nd = dde.grad.jacobian(sigma_voigt_nd, x, i=2, j=0) 

  b_force = (scale.L / scale.sigma) * (-materialData.rho * materialData.g)

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
  # Strain from displacement
  e_x_nd = dde.grad.jacobian(y, x, i=0, j=0)
  e_y_nd = dde.grad.jacobian(y, x, i=1, j=1)
  g_xy_nd = dde.grad.jacobian(y, x, i=0, j=1) + dde.grad.jacobian(y, x, i=1, j=0)
  
  # Voigt notation
  e_voigt_nd = torch.cat([e_x_nd, e_y_nd, g_xy_nd], dim=1)
  e_voigt = e_voigt_nd * (scale.U / scale.L)
  C = materialData.C_stiffness_matrix()
  sigma_voigt_computed = torch.matmul(e_voigt, C) / scale.sigma

  # Stress divergence from predicted stress fields (equilibrium)
  sigmax_x_nd = dde.grad.jacobian(y, x, i=2, j=0)
  sigmay_y_nd = dde.grad.jacobian(y, x, i=3, j=1)
  tauxy_y_nd = dde.grad.jacobian(y, x, i=4, j=1)
  tauxy_x_nd = dde.grad.jacobian(y, x, i=4, j=0)

  # Body force
  b_force_y = (scale.L / scale.sigma) * (-materialData.rho * materialData.g)
  return [
      sigmax_x_nd + tauxy_y_nd,                              # Equilibrium X
      sigmay_y_nd + tauxy_x_nd + b_force_y,                  # Equilibrium Y
      sigma_voigt_computed[:, 0:1] - y[:, 2:3],              # Consistency σ_xx
      sigma_voigt_computed[:, 1:2] - y[:, 3:4],              # Consistency σ_yy  
      sigma_voigt_computed[:, 2:3] - y[:, 4:5]               # Consistency τ_xy
  ]

def pde_2d_ensemble_residual_V2(x, y, scale: EnsemnbleMechanicScale):
    ux = y[:, 0:1]
    uy = y[:, 1:2]
    exx = y[:, 2:3]
    eyy = y[:, 3:4]
    exy = y[:, 4:5]
    sxx = y[:, 5:6]
    syy = y[:, 6:7]
    sxy = y[:, 7:8]

    # Derivatives
    ux_x = dde.grad.jacobian(ux, x, i=0, j=0)
    ux_y = dde.grad.jacobian(ux, x, i=0, j=1)
    uy_x = dde.grad.jacobian(uy, x, i=0, j=0)
    uy_y = dde.grad.jacobian(uy, x, i=0, j=1)

    sxx_x = dde.grad.jacobian(sxx, x, i=0, j=0)
    syy_y = dde.grad.jacobian(syy, x, i=0, j=1)
    sxy_x = dde.grad.jacobian(sxy, x, i=0, j=0)
    sxy_y = dde.grad.jacobian(sxy, x, i=0, j=1)

    # body nd
    fx_nd = 0.0
    fy_nd = (scale.L / scale.sigma) * (-materialData.rho * materialData.g)

    lame_lambda_nd = materialData.lame_lambda / scale.sigma
    lame_mu_nd = materialData.lame_mu / scale.sigma

    # Residuals
    E1 = sxx_x + sxy_y + fx_nd
    E2 = sxy_x + syy_y + fy_nd
    E3 = sxx - lame_lambda_nd * (exx + eyy) - 2 * lame_mu_nd * exx
    E4 = syy - lame_lambda_nd * (exx + eyy) - 2 * lame_mu_nd * eyy
    E5 = sxy - 2 * lame_mu_nd * exy
    E6 = exx - ux_x * (scale.U/scale.L)
    E7 = eyy - uy_y * (scale.U/scale.L)
    E8 = exy - 0.5 * (ux_y + uy_x) * (scale.U/scale.L)

    #print('-----')
    #print('ux', ux.min().item(), ux.max().item())
    #print('uy', uy.min().item(), uy.max().item())
    #print('exx', exx.min().item(), exx.max().item())
    #print('eyy', eyy.min().item(), eyy.max().item())
    #print('exy', exy.min().item(), exy.max().item())
    #print('sxx', sxx.min().item(), sxx.max().item())
    #print('syy', syy.min().item(), syy.max().item())
    #print('sxy', sxy.min().item(), sxy.max().item())

    return [E1, E2, E3, E4, E5, E6, E7, E8]
