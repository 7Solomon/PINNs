from utils.metadata import Domain
from process.heat.scale import *
import numpy as np
import deepxde as dde
from process.heat.residual import lp_residual, steady_lp_residual

def get_steady_domain(domain_vars: Domain, scale: Scale):
    x_min, x_max = domain_vars.spatial['x']
    y_min, y_max = domain_vars.spatial['y']
    geom = dde.geometry.Rectangle((x_min/scale.L, y_min/scale.L), (x_max/scale.L, y_max/scale.L))

    left_value = 100.0 / scale.T
    right_value = 0.0 / scale.T
    bc_left = dde.DirichletBC(geom, lambda x: left_value, 
                            lambda x, on_boundary: on_boundary and np.isclose(x[0], x_min/scale.L))
    bc_right = dde.DirichletBC(geom, lambda x: right_value, 
                            lambda x, on_boundary: on_boundary and np.isclose(x[0], x_max/scale.L))

    data = dde.data.PDE(geom, 
                        lambda x,y: steady_lp_residual(x, y, scale), 
                        [bc_left, bc_right], 
                        num_domain=200, 
                        num_boundary=50)

    return data

def get_transient_domain(domain_vars: Domain, scale: Scale):
    x_min, x_max = domain_vars.spatial['x']
    y_min, y_max = domain_vars.spatial['y']
    t_min, t_max = domain_vars.temporal['t']
    geom = dde.geometry.Rectangle((x_min/scale.L, y_min/scale.L), (x_max/scale.L, y_max/scale.L))

    timedomain = dde.geometry.TimeDomain(t_min/scale.t, t_max/scale.t)   # mit L^2/(pi^2*alpha) geschätzt, und wieder geändert
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    left_value = 100 / scale.T
    right_value = 0 / scale.T
    initial_value = 0 / scale.T
    # BC
    bc_left = dde.DirichletBC(
        geomtime, 
        lambda x: left_value, lambda x, on_boundary: on_boundary and np.isclose(x[0], x_min/scale.L))
    bc_right = dde.DirichletBC(
        geomtime,
        lambda x: right_value, lambda x, on_boundary: on_boundary and np.isclose(x[0], x_max/scale.L))


    # NO FLUX
    bc_top = dde.OperatorBC(
        geomtime, 
        lambda x,y, _: dde.grad.jacobian(y, x, i=0, j=1), lambda x, on_boundary: on_boundary and np.isclose(x[1], y_max/scale.L))
    bc_bottom = dde.OperatorBC(
        geomtime, 
        lambda x,y, _: dde.grad.jacobian(y, x, i=0, j=1), lambda x, on_boundary: on_boundary and np.isclose(x[1], y_min/scale.L))



    # IC
    ic = dde.IC(geomtime, 
        lambda x: initial_value, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime,
        lambda x,y: lp_residual(x, y, scale),
        [ic, bc_left, bc_right, bc_top, bc_bottom],
        num_domain=1600,
        num_boundary=600,
        num_initial=400
    )
    return data
