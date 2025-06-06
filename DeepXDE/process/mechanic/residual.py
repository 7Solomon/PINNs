import math
from utils.functions import voigt_to_tensor
from process.mechanic.scale import Scale
import deepxde as dde
import torch
from config import bernoulliBalkenTConfig, cooksMembranConfig, bernoulliBalken2DConfig, concreteData
    
def pde_1d_residual(x, y):
  w_x = dde.grad.jacobian(y,x, i=0)
  w_xx = dde.grad.jacobian(w_x, x, i=0)
  w_xxx = dde.grad.jacobian(w_xx, x, i=0)
  w_xxxx = dde.grad.jacobian(w_xxx, x, i=0)
  return w_xxxx - 1.0
  #return bernoulliBalkenConfig.EI*w_xxxx - bernoulliBalkenConfig.f(x[:,0], x[:,1])
def pde_2d_residual(x, y, scale: Scale):
  e_x = dde.grad.jacobian(y,x, i=0, j=0) * (scale.U(concreteData.E)/ scale.L)
  e_y = dde.grad.jacobian(y,x, i=1, j=1) * (scale.U(concreteData.E) / scale.L)
  g_xy = dde.grad.jacobian(y,x, i=0, j=1) * (scale.U(concreteData.E) / scale.L) + dde.grad.jacobian(y,x, i=1, j=0) * (scale.U(concreteData.E) / scale.L)
  voigt = torch.cat([e_x, e_y, g_xy], dim=1)
  C_scaled = bernoulliBalken2DConfig.C(concreteData) /scale.sigma(concreteData.E)  # [N/L**2]
  sigma_voigt = torch.matmul(voigt, C_scaled)   # [1/L**2]

  sigmax_x = dde.grad.jacobian(sigma_voigt, x, i=0, j=0) * (scale.sigma(concreteData.E) / scale.L)   # [N/L**2] / [L] = [N/L]
  sigmay_y = dde.grad.jacobian(sigma_voigt, x, i=1, j=1) * (scale.sigma(concreteData.E) / scale.L)   # [N/L]
  tauxy_y = dde.grad.jacobian(sigma_voigt, x, i=2, j=1) * (scale.sigma(concreteData.E) / scale.L)    # [N/L]
  tauxy_x = dde.grad.jacobian(sigma_voigt, x, i=2, j=0) * (scale.sigma(concreteData.E) / scale.L)    # [N/L]

  #print('u', y[0].min().item(), y[0].max().item())
  #print('v', y[1].min().item(), y[1].max().item())
  #print('e_x', e_x.min().item(), e_x.max().item())
  #print('e_y', e_y.min().item(), e_y.max().item())
  #print('g_xy', g_xy.min().item(), g_xy.max().item())
  #print('sigmax_x', sigmax_x.min().item(), sigmax_x.max().item())
  #print('sigmay_y', sigmay_y.min().item(), sigmay_y.max().item())
  #print('tauxy_y', tauxy_y.min().item(), tauxy_y.max().item())
  #print('tauxy_x', tauxy_x.min().item(), tauxy_x.max().item())

  return [sigmax_x + tauxy_y, sigmay_y + tauxy_x - 1.0]
def pde_1d_t_residual(x, y): 
  w_tt = dde.grad.hessian(y,x, i=1,j=1)
  w_xx = dde.grad.hessian(y,x, i=0,j=0)
  w_xxxx = dde.grad.hessian(w_xx,x, i=0,j=0)
  return w_tt + w_xxxx - bernoulliBalkenTConfig.f(x[:,0], x[:,1])
  #return 10**2*w_tt + 4 * 10**6*w_xxxx - 4 * 10**6*(1-16*math.pi**2)*math.sin(x[0]/math.pi)*math.cos(800*x[1]/math.pi)/math.pi**3
  #return w_tt - bernoulliBalkenTConfig.EI*w_xxxx - bernoulliBalkenTConfig.f(x[:,0], x[:,1])
def calc_sigma(x,y):
  pass

def cooks_residual(x,y):

  du_dx = dde.grad.jacobian(y, x, i=0, j=0)
  dv_dy = dde.grad.jacobian(y, x, i=1, j=1)
  du_dy = dde.grad.jacobian(y, x, i=0, j=1)
  dv_dx = dde.grad.jacobian(y, x, i=1, j=0)

  e_voigt = torch.cat([du_dx, dv_dy, du_dy + dv_dx], dim=1)
  sigma_voigt = torch.matmul(e_voigt, cooksMembranConfig.C())

  sigma = voigt_to_tensor(sigma_voigt)
  sigma = scale_u(sigma)

  dsigma_dx = dde.grad.jacobian(sigma, x, i=0)
  dsigma_dy = dde.grad.jacobian(sigma, x, i=1)

  return dsigma_dx + dsigma_dy # + cooksMembranConfig.f(x)
