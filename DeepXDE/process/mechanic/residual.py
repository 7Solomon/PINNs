import math
from utils.function_utils import voigt_to_tensor
from process.mechanic.scale import *
import deepxde as dde
import torch
from config import bernoulliBalkenTConfig, cooksMembranConfig
    
def pde_1d_residual(x, y):
  w_x = dde.grad.jacobian(y,x, i=0)
  w_xx = dde.grad.jacobian(w_x, x, i=0)
  w_xxx = dde.grad.jacobian(w_xx, x, i=0)
  w_xxxx = dde.grad.jacobian(w_xxx, x, i=0)
  return w_xxxx - 1.0
  #return bernoulliBalkenConfig.EI*w_xxxx - bernoulliBalkenConfig.f(x[:,0], x[:,1])

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
