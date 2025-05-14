import math
import numpy as np
import deepxde as dde
from process.mechanic.residual import pde_1d_residual, pde_1d_t_residual

def scale_value(T):
    return T/100


def du_dxx_zero(x, y, _):
    return dde.grad.hessian(y, x)[:,0]
def du_dxx_one(x, y, _):
    return dde.grad.hessian(y, x)[:,0] - 1.0

def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)
def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)
def get_fest_los_domain():
    geom = dde.geometry.Interval(0,1)

    bc_left_w = dde.DirichletBC(geom, lambda x: 0, boundary_left)
    bc_right_w = dde.DirichletBC(geom, lambda x: 0, boundary_right)

    bc_left_wxx = dde.OperatorBC(geom, du_dxx_zero, boundary_left)
    bc_right_wxx = dde.OperatorBC(geom, du_dxx_zero, boundary_right)
    
    data = dde.data.PDE(geom, 
                        pde_1d_residual, 
                        [bc_left_w, bc_right_w, bc_left_wxx, bc_right_wxx], 
                        num_domain=200, 
                        num_boundary=50)
    # wights 
    #data.set_weights([1, 1, 1, 1])
    return data

def get_einspannung_domain():
    geom = dde.geometry.Interval(0,1)

    bc_left_w = dde.DirichletBC(geom, lambda x: 0, boundary_left)
    bc_left_wx = dde.OperatorBC(geom, 
                                lambda x,y,_: dde.grad.jacobian(y, x)[:,0], 
                                boundary_left)

    bc_right_wxx = dde.OperatorBC(geom, 
                                lambda x,y,_: dde.grad.jacobian(y, x)[:,0] - 1.0,
                                boundary_right)
    bc_right_wxxx = dde.OperatorBC(geom, 
                                lambda x, y, _: dde.grad.jacobian(dde.grad.hessian(y, x), x)[:,0] - 1.0,
                                boundary_right)
    data = dde.data.PDE(geom,
                        pde_1d_residual, 
                        [bc_left_w, bc_left_wx, bc_right_wxx, bc_right_wxxx], 
                        num_domain=200, 
                        num_boundary=50)
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

def cooks_right(x, _):
    return np.isclose(x[0], 48)

def get_cooks_domain():
    geom = dde.geometry.Polygon([
        [0, 0],
        [48, 44],
        [48, 60],
        [0, 44],
    ])
    time = dde.geometry.TimeDomain(0, 1)
    geomTime = dde.geometry.GeometryXTime(geom, time)

    bc_left_w_x = dde.DirichletBC(geomTime, lambda x,y: 0, boundary_left, component=0)
    bc_left_w_y = dde.DirichletBC(geomTime, lambda x,y: 0, boundary_left, component=1)

    bc_right_w_x = dde.DirichletBC(geomTime, lambda x,y: 0, boundary_left, component=0)
    bc_right_w_xx = dde.OperatorBC(geomTime, lambda x,y: dde.grad.jacobian(dde.grad.hessian(y, x)[:,0], x)[:,0] - 1.0, cooks_right, component=0)
    data = dde.data.TimePDE(geomTime,
                            pde_1d_t_residual, 
                            [bc_left_w_x, bc_left_w_y, bc_right_w_x, bc_right_w_xx], 
                            num_domain=1000, 
                            num_boundary=400,
                            num_initial=200)

def get_domain(type):
    if type == 'fest_los':
        return get_fest_los_domain()
    elif type == 'einspannung':
        return get_einspannung_domain()	
    elif type == 'fest_los_t':
        return get_fest_los_t_domain()
    else:
        raise ValueError(f"Unknown domain type: {type}")