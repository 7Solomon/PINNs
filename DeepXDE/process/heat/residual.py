import deepxde as dde
from config import transientHeatConfig

def lp_residual(x,y):
    y_t=dde.grad.jacobian(y, x, i=0)[2]
    y_xx=dde.grad.hessian(y, x)
    y_yy=dde.grad.hessian(y, x, i=1)
    return y_t - transientHeatConfig.alpha * (y_xx + y_yy)


def steady_lp_residual(x, y):
    y_xx=dde.grad.hessian(y, x)
    y_yy=dde.grad.hessian(y, x, i=1)
    return y_xx + y_yy