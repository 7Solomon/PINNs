from process.heat.scale import *
import deepxde as dde
from config import concreteData

def lp_residual(x,y, transient_heat_scaling):
    T_t=dde.grad.jacobian(y, x, i=0, j=2)
    T_xx=dde.grad.hessian(y, x, i=0, j=0)
    T_yy=dde.grad.hessian(y, x, i=0, j=1)   
    return T_t - (transient_heat_scaling.scale_alpha_x(concreteData.alpha())*T_xx + transient_heat_scaling.scale_alpha_y(concreteData.alpha())*T_yy)


def steady_lp_residual(x, y, steady_heat_scaling):
    T_xx=dde.grad.hessian(y, x, i=0)
    T_yy=dde.grad.hessian(y, x, i=1)
    return T_xx + T_yy
