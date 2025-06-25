from process.heat.scale import *
import deepxde as dde
from material import concreteData

def lp_residual(x,y, scale: Scale): 
    T_t=dde.grad.jacobian(y, x, i=0, j=2)
    T_xx=dde.grad.hessian(y, x, i=0, j=0)
    T_yy=dde.grad.hessian(y, x, i=0, j=1)
    alpha = concreteData.alpha_thermal_diffusivity

    pi_one = scale.t / scale.L**2 # [-]
    #print('----')
    #print('T_t', T_t.min().item(), T_t.max().item())
    #print('T_xx', T_xx.min().item(), T_xx.max().item())
    #print('T_yy', T_yy.min().item(), T_yy.max().item())
    #print('alpha', alpha)
    #print('pi_one', pi_one)
    #print('scale.L', scale.L)
    #print('scale.t', scale.t)
    #print('scale.alpha', scale.alpha)
    #print('scale.T', scale.T)
    #print('scale.Temperature', scale.T)
    return T_t - alpha * (T_xx + T_yy) * pi_one


def steady_lp_residual(x, y, scale: Scale):
    T_xx=dde.grad.hessian(y, x, i=0)
    T_yy=dde.grad.hessian(y, x, i=1)
    # kein Scaling weil kein Q
    return T_xx + T_yy
