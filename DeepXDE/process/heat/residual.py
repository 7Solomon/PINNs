from process.heat.scale import *
import deepxde as dde
from config import concreteData

def lp_residual(x,y, scale: Scale): 
    T_t=dde.grad.jacobian(y, x, i=0, j=2)
    T_xx=dde.grad.hessian(y, x, i=0, j=0)
    T_yy=dde.grad.hessian(y, x, i=0, j=1)
    alpha = concreteData.alpha()
    return T_t - (alpha*scale.alpha_x * T_xx + alpha*scale.alpha_y * T_yy)


def steady_lp_residual(x, y, scale: Scale):
    T_xx=dde.grad.hessian(y, x, i=0)
    T_yy=dde.grad.hessian(y, x, i=1)
    return T_xx + T_yy
