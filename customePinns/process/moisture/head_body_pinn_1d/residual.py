import torch
from nn_stuff.pinn import BodyHeadPINN

def predicted_K_theta_residual(model: BodyHeadPINN, x, debug_print=False):
    x = x.clone().detach().requires_grad_(True)
    psi, K, theta  = model(x)

    K_z = torch.autograd.grad(K,x, grad_outputs=torch.ones_like(K), create_graph=True)[0][:, 0]

    psi_z = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0][:, 0]
    psi_z_z = torch.autograd.grad(psi_z, x, grad_outputs=torch.ones_like(psi_z), create_graph=True)[0][:, 0]
    
    theta_t = torch.autograd.grad(theta, x, grad_outputs=torch.ones_like(theta), create_graph=True)[0][:, 1]
    
    if debug_print:
        print('-------------')
        print(f'psi: {psi[:2].detach().cpu().numpy()}')
        print(f'K: {K[:2].detach().cpu().numpy()}')
        print(f'theta: {theta[:2].detach().cpu().numpy()}')
        print('---')
        print(f'K_z: {K_z[:2].detach().cpu().numpy()}')
        print(f'psi_z: {psi_z[:2].detach().cpu().numpy()}')
        print(f'psi_z_z: {psi_z_z[:2].detach().cpu().numpy()}')
        print(f'theta_t: {theta_t[:2].detach().cpu().numpy()}')

    return theta_t - K_z*psi_z - K*psi_z_z - K_z