import torch
from vars import alpha

def lp_residual(model, x: torch.Tensor):
    assert x.dim() == 3, f'3D tensor erwartet, aber ist {x.dim()}D tensor'
    T = model(x)

    grads = torch.autograd.grad(T,x,grad_outputs=torch.ones_like(T),create_graph=True)[0]
    T_x = grads[:,0]
    T_y = grads[:,1]
    T_t = grads[:,2]

    T_xx = torch.autograd.grad(T_x,x,grad_outputs=torch.ones_like(T_x),create_graph=True)[0][:,0]
    T_yy = torch.autograd.grad(T_y,x,grad_outputs=torch.ones_like(T_y),create_graph=True)[0][:,1]

    return T_t - alpha * (T_xx + T_yy)

def steady_lp_residual(model, x):
  assert x.dim() == 2, f'2D tensor erwartet, aber ist {x.dim()}D tensor'
  T = model(x)

  grads = torch.autograd.grad(T,x,grad_outputs=torch.ones_like(T),create_graph=True)[0]
  T_x = grads[:,0]
  T_y = grads[:,1]

  T_xx = torch.autograd.grad(T_x,x,grad_outputs=torch.ones_like(T_x),create_graph=True)[0][:,0]
  T_yy = torch.autograd.grad(T_y,x,grad_outputs=torch.ones_like(T_y),create_graph=True)[0][:,1]

  return T_xx + T_yy