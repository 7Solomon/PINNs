from process.thermal_moisture.scale import Scale
from config import concreteData
import deepxde as dde
import torch

def c_eff(theta, T):  # for sml range of T, is not dependent on T
    return (concreteData.rho*concreteData.cp*(1-concreteData.phi) 
        + theta*concreteData.rho_w*concreteData.cp_w
        + (concreteData.phi - theta)*concreteData.rho_a*concreteData.cp_a
    )

def thermal_conductivity(theta, T):  # again T little influence, but can be changed
    return (concreteData.lamda_dry
        - (concreteData.lamda_sat - concreteData.lamda_dry) * (theta/concreteData.theta_s)**concreteData.n
    )

def residual(x,y,scale: Scale):
    T_phys = y[:,0:1] * scale.Temperature
    theta_phys = y[:,1:2] * scale.theta
    theta_phys_pos = torch.clamp(theta_phys, min=1e-8, max=concreteData.theta_s - 1e-8)

    ceff = c_eff(theta_phys_pos, T_phys)  / scale.c0
    dT_dt = dde.grad.jacobian(y, x, i=0, j=2)
    time_term = ceff * dT_dt
    

    lamda = thermal_conductivity(theta_phys_pos, T_phys) / scale.lamda
    dT_dx = dde.grad.jacobian(y, x, i=0, j=0)
    dT_dy = dde.grad.jacobian(y, x, i=0, j=1)
    grad_T = torch.cat([dT_dx,dT_dy], dim=1)  # [N, 2]
    lamda_grad_T = lamda * grad_T
    pi_one = scale.lamda * scale.t / (scale.L**2 * scale.c0)
    heat_conduction = pi_one * (dde.grad.jacobian(lamda_grad_T, x, i=0, j=0) + dde.grad.jacobian(lamda_grad_T, x, i=1, j=1))

    d_theta_dt = dde.grad.jacobian(y, x, i=1, j=2)
    pi_two = scale.theta / (scale.c0 * scale.Temperature) * concreteData.L_v  # HERe Added LV to make pitwo  some good stuff
    latent_heat_through_m_change = pi_two  * d_theta_dt

    #print('------')
    #print(f'c_eff:  {ceff.min().item()}, {ceff.max().item()}')
    #print(f'dT_dt: {dT_dt.min().item()}, {dT_dt.max().item()}')
    #print(f'time_term: {time_term.min().item()}, {time_term.max().item()}')
    #print(f'lamda: {lamda.min().item()}, {lamda.max().item()}')
    #print(f'dT_dx: {dT_dx.min().item()}, {dT_dx.max().item()}')
    #print(f'dT_dy:  {dT_dy.min().item()}, {dT_dy.max().item()}')
    #print(f'grad_T: {grad_T.min().item()}, {grad_T.max().item()}')
    #print(f'pi_one: {pi_one}')
    #print(f'lamda_grad_T:  {lamda_grad_T.min().item()}, {lamda_grad_T.max().item()}')
    #print(f'heat_conduction: {heat_conduction.min().item()}, {heat_conduction.max().item()}')
    #print(f'd_theta_dt: {d_theta_dt.min().item()}, {d_theta_dt.max().item()}')
    #print(f'latent_heat_through_m_change: {latent_heat_through_m_change.min().item()}, {latent_heat_through_m_change.max().item()}')
    #print(f'pi_two: {pi_two}')
    #print(f'scale.L: {scale.L}, scale.t: {scale.t}, scale.Temperature: {scale.Temperature}, scale.theta: {scale.theta}')
    #print(f'scale.c0: {scale.c0}, scale.lamda: {scale.lamda}')
    #print(f'concreteData.L_v: {concreteData.L_v}, concreteData.rho_w: {concreteData.rho_w}, concreteData.cp_w: {concreteData.cp_w}')
    return heat_conduction + latent_heat_through_m_change - time_term