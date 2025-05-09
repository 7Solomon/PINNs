import deepxde as dde

def pde_residual(x, y):
  w_x = dde.grad.jacobian(y,x, i=0)
  w_xx = dde.grad.jacobian(w_x, x, i=0)
  w_xxx = dde.grad.jacobian(w_xx, x, i=0)
  w_xxxx = dde.grad.jacobian(w_xxx, x, i=0)
  return w_xxxx - 1.0