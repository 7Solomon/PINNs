import torch
from utils.metadata import Domain
from process.mechanical_moisture.residual import moisture_diffusivity, residual
from process.mechanical_moisture.scale import Scale
import deepxde as dde
import numpy as np

from material import concreteData
materialData = concreteData

#def top_flux_function(x, y, scale: Scale):
#
#    q_in = 1e-5 / scale.q
#
#
#    e_x = dde.grad.jacobian(y, x, i=0, j=0)
#    e_y = dde.grad.jacobian(y,x, i=1, j=1)
#    g_xy = dde.grad.jacobian(y,x, i=0, j=1) + dde.grad.jacobian(y,x, i=1, j=0)
#    e_voigt = torch.cat([e_x, e_y, g_xy], dim=1)
#    
#    phys_e_voigt = e_voigt * scale.epsilon 
#    phys_theta = y[:,2:3] * scale.theta
#    D = moisture_diffusivity(phys_theta, phys_e_voigt) / (scale.L**2/scale.t)
#    flux = -q_in / D
#    return flux 


def domain_2d(domain_vars: Domain, scale: Scale):
    x_min, x_max = domain_vars.spatial['x']
    y_min, y_max = domain_vars.spatial['y']
    t_min, t_max = domain_vars.temporal['t']
   
    geom = dde.geometry.Rectangle(xmin=[x_min/scale.L, y_min/scale.L], xmax=[x_max/scale.L, y_max/scale.L])
    time = dde.geometry.TimeDomain(t_min/scale.t, t_max/scale.t)
    geom_time = dde.geometry.GeometryXTime(geom, time)

    bc_left_u = dde.DirichletBC(geom_time, lambda x: 0.0, 
                                       lambda x, _ :_ and np.isclose(x[0], x_min/scale.L) , component=0)
    bc_left_v = dde.DirichletBC(geom_time, lambda x: 0.0, 
                                       lambda x, _ :_ and np.isclose(x[0], x_min/scale.L), component=1)
    theta_bottom_val = materialData.theta_r
    theta_top_val = 0.9 * materialData.theta_s
    bc_top_theta = dde.DirichletBC(geom_time, lambda x: theta_top_val,
                                       lambda x, _ :_ and np.isclose(x[1], y_max/scale.L), component=2)
    
    def initial_theta_smooth(x):
        y_normalized = (x[1] - y_min) / (y_max - y_min)
        return theta_bottom_val + (theta_top_val - theta_bottom_val) * y_normalized
    ic_theta = dde.IC(geom_time, 
                        initial_theta_smooth, 
                        lambda _, on_initial: on_initial, component=2)

    data = dde.data.TimePDE(geom_time,
                        lambda x,y : residual(x, y, scale), 
                        [ic_theta, bc_left_u, bc_left_v, bc_top_theta], 
                        num_domain=2000, 
                        num_boundary=500,
                        num_initial=500)
    return data