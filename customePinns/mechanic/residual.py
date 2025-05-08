import torch

def residual(w ,conf, x: torch.Tensor):
    """
    x[0] needs to be x coords, x[1] needs to be q(x)
    """

    w_x = torch.autograd.grad(w,x,grad_outputs=torch.ones_like(w),create_graph=True)[0][:,0]
    w_xx = torch.autograd.grad(w_x,x,grad_outputs=torch.ones_like(w_x),create_graph=True)[0][:,0]
    w_xxx = torch.autograd.grad(w_xx,x,grad_outputs=torch.ones_like(w_xx),create_graph=True)[0][:,0]
    w_xxxx = torch.autograd.grad(w_xxx,x,grad_outputs=torch.ones_like(w_xxx),create_graph=True)[0][:,0]
    return conf.EI*w_xxxx - x[:,1]