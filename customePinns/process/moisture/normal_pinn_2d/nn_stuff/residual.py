from nn_stuff.pinn import BodyHeadPINN
from moisture.normal_pinn_2d.physics import *
import torch

def residual(model, x, conf):
    u_pred = model(x)  # u is pressure head

    theta = WRC(u_pred, conf)
    K = HC(u_pred, conf)
    
    #print(f'K: {K[0][0]}')
    #print(f'theta: {theta[0][0]}')
    
    theta_t = torch.autograd.grad(theta, x, grad_outputs=torch.ones_like(theta), create_graph=True)[0][:,2]

    grad_h = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(theta), create_graph=True)[0]  # Alle Gradienten nach x, y, t
    h_x = grad_h[:, 0]
    h_y = grad_h[:, 1]

    K_h_x = K  * h_x
    K_h_y = K  * h_y

    d_Kh_dx = torch.autograd.grad(K_h_x, x, grad_outputs=torch.ones_like(K_h_x), create_graph=True)[0][:, 0]
    d_Kh_dy = torch.autograd.grad(K_h_y, x, grad_outputs=torch.ones_like(K_h_y), create_graph=True)[0][:, 1]
    
    div_K_grad_h = d_Kh_dx + d_Kh_dy
    return theta_t - div_K_grad_h


def test_split_residual(model, x, conf):
    u_pred = model(x)  # u is pressure head

    theta = WRC(u_pred, conf)
    K = HC(u_pred, conf)
    
    theta_t = torch.autograd.grad(theta, x, grad_outputs=torch.ones_like(theta), create_graph=True)[0][:,2]

    grad_h = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(theta), create_graph=True)[0]  # Alle Gradienten nach x, y, t
    h_x = grad_h[:, 0]
    h_y = grad_h[:, 1]

    K_h_x = K  * h_x
    K_h_y = K  * h_y

    d_Kh_dx = torch.autograd.grad(K_h_x, x, grad_outputs=torch.ones_like(K_h_x), create_graph=True)[0][:, 0]
    d_Kh_dy = torch.autograd.grad(K_h_y, x, grad_outputs=torch.ones_like(K_h_y), create_graph=True)[0][:, 1]
    
    div_K_grad_h = d_Kh_dx + d_Kh_dy

    loss_time = torch.mean(theta_t**2)
    loss_space = torch.mean(div_K_grad_h**2)
    return theta_t - div_K_grad_h



def test_log_residual(model, x):
    h = model(x)
    theta = WRC(h)
    
    logK = torch.log10(HC(h)+ 1e-15) # epsilon stuff to not make log(0)
    grad_logK = torch.autograd.grad(logK, x, grad_outputs=torch.ones_like(logK), create_graph=True)[0]
    h_x = grad_logK[:, 0]
    h_y = grad_logK[:, 1]

    K = torch.exp(logK)

    laplace_h_x = torch.autograd.grad(h_x, x, grad_outputs=torch.ones_like(h_x), create_graph=True)[0][:, 0]
    laplace_h_y = torch.autograd.grad(h_y, x, grad_outputs=torch.ones_like(h_y), create_graph=True)[0][:, 1]
    laplace_h = laplace_h_x + laplace_h_y

    dot_grad_log_K_grad_h = grad_logK[:, 0] * h_x + grad_logK[:, 1] * h_y
    div_K_grad_h = K*(laplace_h + dot_grad_log_K_grad_h)
    theta_t = torch.autograd.grad(theta, x, grad_outputs=torch.ones_like(theta), create_graph=True)[0][:,2]
    
    return theta_t - div_K_grad_h
