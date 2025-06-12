from utils.metadata import Domain
from process.heat.scale import *
import numpy as np
import deepxde as dde
from process.heat.residual import lp_residual, steady_lp_residual

def get_steady_domain(domain_vars):
    x_min, x_max = domain_vars.spatial['x']
    y_min, y_max = domain_vars.spatial['y']
    scale = Scale(domain_vars)
    geom = dde.geometry.Rectangle((x_min/scale.Lx, y_min/scale.Ly), (x_max/scale.Lx, y_max/scale.Ly))

    left_value = 100.0 / scale.T
    right_value = 0.0 / scale.T
    bc_left = dde.DirichletBC(geom, lambda x: left_value, 
                            lambda x, on_boundary: on_boundary and np.isclose(x[0], x_min/scale))
    bc_right = dde.DirichletBC(geom, lambda x: right_value, 
                            lambda x, on_boundary: on_boundary and np.isclose(x[0], x_max/scale))

    data = dde.data.PDE(geom, 
                        lambda x,y: steady_lp_residual(x, y, scale), 
                        [bc_left, bc_right], 
                        num_domain=200, 
                        num_boundary=50)
    # wights 
    #data.set_weights([1, 1, 1, 1])
    
    return data

def get_transient_domain(domain_vars):
    x_min, x_max = domain_vars.spatial['x']
    y_min, y_max = domain_vars.spatial['y']
    t_min, t_max = domain_vars.temporal['t']
    scale = Scale(domain_vars)
    geom = dde.geometry.Rectangle((x_min/scale.Lx, y_min/scale.Ly), (x_max/scale.Lx, y_max/scale.Ly))

    timedomain = dde.geometry.TimeDomain(t_min/scale.t, t_max/scale.t)   # mit L^2/(pi^2*alpha) geschätzt, und wieder geändert
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    right_value = 100 / scale.T
    left_value = 0 / scale.T
    initial_value = 0 / scale.T
    # BC
    bc_left = dde.DirichletBC(
        geomtime, 
        lambda x: left_value, lambda x, on_boundary: on_boundary and np.isclose(x[0], x_min/scale.Lx))
    bc_right = dde.DirichletBC(
        geomtime,
        lambda x: right_value, lambda x, on_boundary: on_boundary and np.isclose(x[0], x_max/scale.Lx))

    # IC
    ic = dde.IC(geomtime, 
        lambda x: initial_value, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime,
        lambda x,y: lp_residual(x, y, scale),
        [bc_left, bc_right, ic],
        num_domain=1600,
        num_boundary=600,
        num_initial=400
    )
    return data
