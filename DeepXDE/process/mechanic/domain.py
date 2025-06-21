import math
from utils.metadata import Domain
from utils.functions import voigt_to_tensor
from process.mechanic.scale import *
import numpy as np
import deepxde as dde
import torch
from process.mechanic.residual import pde_1d_residual, pde_1d_t_residual, pde_2d_residual
from config import cooksMembranConfig

def du_dxx_zero(x, y, _):
    return dde.grad.hessian(y, x)[:,0]
def du_dxx_one(x, y, _):
    return dde.grad.hessian(y, x)[:,0] - 1.0

def get_sigma_voigt_nd(x, y, scale: Scale):
    e_voigt_nd = torch.cat([dde.grad.jacobian(y,x, i=0, j=0), 
                            dde.grad.jacobian(y,x, i=1, j=1), 
                            dde.grad.jacobian(y,x, i=0, j=1) + dde.grad.jacobian(y,x, i=1, j=0)
                            ], dim=1)
    e_voigt  = e_voigt_nd * (scale.U / scale.L)
    C = materialData.C_stiffness_matrix()
    sigma_voigt_nd = torch.matmul(e_voigt, C) / scale.sigma 
    return sigma_voigt_nd

def get_fest_los_domain(domain_vars: Domain, scale: Scale):
    x_min, x_max = domain_vars.spatial['x']
    geom = dde.geometry.Interval(x_min, x_max)
    bc_left_w = dde.DirichletBC(geom, lambda x: 0, lambda x, _ :_ and np.isclose(x[0], x_min))
    bc_right_w = dde.DirichletBC(geom, lambda x: 0, lambda x, _ :_ and np.isclose(x[0], x_max))
    bc_left_wxx = dde.OperatorBC(geom, du_dxx_zero, lambda x, _ :_ and np.isclose(x[0], x_min))
    bc_right_wxx = dde.OperatorBC(geom, du_dxx_zero, lambda x, _ :_ and np.isclose(x[0], x_max))
    data = dde.data.PDE(geom,
                        pde_1d_residual,
                        [bc_left_w, bc_right_w, bc_left_wxx, bc_right_wxx],
                        num_domain=200,
                        num_boundary=50)
    # wights 
    #data.set_weights([1, 1, 1, 1])
    return data

def get_einspannung_domain_2d(domain_vars: Domain, scale: Scale):
    x_min, x_max = domain_vars.spatial['x']
    y_min, y_max = domain_vars.spatial['y']
    geom = dde.geometry.Rectangle(xmin=[x_min/scale.L, y_min/scale.L], xmax=[x_max/scale.L, y_max/scale.L])

    bc_left_u_x = dde.DirichletBC(geom, lambda x: 0.0, 
                                       lambda x, _ :_ and np.isclose(x[0], x_min/scale.L), component=0)
    bc_left_u_y = dde.DirichletBC(geom, lambda x: 0.0, 
                                       lambda x, _ :_ and np.isclose(x[0], x_min/scale.L), component=1)

    # traction-free boundary at right (sigma_xx=0, sigma_xy=0)
    #bc_right_u_xx = dde.OperatorBC(geom,
    #                            lambda x,y,_: get_sigma_voigt_nd(x, y, scale)[:,0:1],
    #                            lambda x, _ :_ and np.isclose(x[0], x_max/scale.L))
    #bc_right_u_yxx = dde.OperatorBC(geom,
    #                            lambda x,y,_: get_sigma_voigt_nd(x, y, scale)[:,1:2],
    #                            lambda x, _ :_ and np.isclose(x[0], x_max/scale.L))
    
    
    # traction-free boundary at top (sigma_yy=0, tau_xy=0)
    #bc_top_sigma_yy = dde.OperatorBC(geom,
    #                                 lambda x, y, _: get_sigma_voigt_nd(x, y, scale)[:, 1:2],
    #                                 lambda x, on_boundary: on_boundary and np.isclose(x[1], y_max/scale.L))
    #bc_top_tau_xy = dde.OperatorBC(geom,
    #                               lambda x, y, _: get_sigma_voigt_nd(x, y, scale)[:, 2:3],
    #                               lambda x, on_boundary: on_boundary and np.isclose(x[1], y_max/scale.L))
    ## traction-free boundary at bottom (sigma_yy=0, tau_xy=0)
    #bc_bottom_sigma_yy = dde.OperatorBC(geom,
    #                                    lambda x, y, _: get_sigma_voigt_nd(x, y, scale)[:, 1:2],
    #                                    lambda x, on_boundary: on_boundary and np.isclose(x[1], y_min/scale.L))
    #bc_bottom_tau_xy = dde.OperatorBC(geom,
    #                                  lambda x, y, _: get_sigma_voigt_nd(x, y, scale)[:, 2:3],
    #                                  lambda x, on_boundary: on_boundary and np.isclose(x[1], y_min/scale.L))

    data = dde.data.PDE(geom,
                        lambda x,y : pde_2d_residual(x, y, scale),
                        [bc_left_u_x, bc_left_u_y], #,bc_right_u_xx, bc_right_u_yxx, bc_top_sigma_yy, bc_top_tau_xy, bc_bottom_sigma_yy, bc_bottom_tau_xy],
                        num_domain=2000,
                        num_boundary=500)
    return data

def get_einspannung_domain(domain_vars: Domain, scale: Scale):
    x_min, x_max = domain_vars.spatial['x']
    geom = dde.geometry.Interval(x_min, x_max)

    bc_left_w = dde.DirichletBC(geom, lambda x: 0, lambda x, _ :_ and np.isclose(x[0], x_min))
    bc_left_wx = dde.OperatorBC(geom, 
                                lambda x,y,_: dde.grad.jacobian(y, x)[:,0], 
                                lambda x, _ :_ and np.isclose(x[0], x_min))

    bc_right_wxx = dde.OperatorBC(geom, 
                                lambda x,y,_: dde.grad.jacobian(y, x)[:,0] - 1.0,
                                lambda x, _ :_ and np.isclose(x[0], x_max))
    bc_right_wxxx = dde.OperatorBC(geom, 
                                lambda x, y, _: dde.grad.jacobian(dde.grad.hessian(y, x), x)[:,0] - 1.0,
                                lambda x, _ :_ and np.isclose(x[0], x_max))
    data = dde.data.PDE(geom,
                        pde_1d_residual, 
                        [bc_left_w, bc_left_wx, bc_right_wxx, bc_right_wxxx], 
                        num_domain=2000, 
                        num_boundary=1000)
    return data

def boundary_left_time(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)
def boundary_right_time(x, on_boundary):
    return on_boundary and np.isclose(x[0], math.pi)
def u_t_zero(x, y, _):
    return dde.grad.jacobian(y, x, i=0)[:,1]
def u_xx_zero(x, y, _):
    return dde.grad.hessian(y, x)[:,0]

def get_fest_los_t_domain():
    geom = dde.geometry.Interval(0,math.pi)
    time = dde.geometry.TimeDomain(0,1)
    geotime = dde.geometry.GeometryXTime(geom, time)

    init_dir = dde.IC(geotime, lambda x: np.sin(x[:,0]), lambda _, on_initial: on_initial)
    init_neu = dde.OperatorBC(geotime, u_t_zero, lambda _, on_initial: on_initial)
    
    bc_left = dde.DirichletBC(geotime, lambda x: 0.0, boundary_left_time)
    bc_left_xx = dde.OperatorBC(geotime, u_xx_zero, boundary_left_time)
    bc_right_xx = dde.OperatorBC(geotime, u_xx_zero, boundary_right_time)

    data = dde.data.TimePDE(geotime,
                            pde_1d_t_residual, 
                            [bc_left, bc_left_xx, bc_right_xx, init_dir, init_neu], 
                            num_domain=1000, 
                            num_boundary=400,
                            num_initial=200)
    return data

#def cooks_right(x, on_boundary):
#    return on_boundary and np.isclose(x[0], scale_x(48.0))
#def cooks_left(x, on_boundary):
#    return on_boundary and np.isclose(x[0], scale_x(0.0))
#def cooks_right_value(x, y, _):
#    du_dx = dde.grad.jacobian(y, x, i=0, j=0)
#    dv_dy = dde.grad.jacobian(y, x, i=1, j=1)
#    du_dy = dde.grad.jacobian(y, x, i=0, j=1)
#    dv_dx = dde.grad.jacobian(y, x, i=1, j=0)
#
#    e_voigt = torch.cat([du_dx, dv_dy, du_dy + dv_dx], dim=1)
#    sigma_voigt = torch.matmul(e_voigt, cooksMembranConfig.C())
#
#    sigma = voigt_to_tensor(sigma_voigt)
#    sigma = scale_u(sigma)
#    return sigma[:, 1, 1] - scale_f(20.0)
#
def get_cooks_domain():
    raise NotImplementedError('DU KEK')
#    geom = dde.geometry.Polygon([
#        [scale_x(0), scale_x(0)],
#        [scale_x(48), scale_x(44)],
#        [scale_x(48), scale_x(60)],
#        [scale_x(0), scale_x(44)],
#    ])
#    #time = dde.geometry.TimeDomain(0, 1)
#    #geomTime = dde.geometry.GeometryXTime(geom, time)
#
#    bc_left_w_x = dde.DirichletBC(geom, lambda x: 0, cooks_left, component=0)
#    bc_left_w_y = dde.DirichletBC(geom, lambda x: 0, cooks_left, component=1)
#
#    bc_right_w_xx = dde.OperatorBC(geom, cooks_right_value, cooks_right)
#    data = dde.data.PDE(geom,
#                            cooks_residual, 
#                            [bc_left_w_x, bc_left_w_y, bc_right_w_xx], 
#                            num_domain=100, 
#                            num_boundary=40,
#                            )
#    return data
