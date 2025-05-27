from utils.metadata import Domain
from process.heat.scale import *
import numpy as np
import deepxde as dde
from process.heat.residual import lp_residual, steady_lp_residual


def boundary_left(x, on_boundary, x_min, transient_heat_scaling):
    return on_boundary and np.isclose(x[0], transient_heat_scaling.scale_x(x_min))
def boundary_right(x, on_boundary, x_max, transient_heat_scaling):
    return on_boundary and np.isclose(x[0], transient_heat_scaling.scale_x(x_max))

def get_steady_domain(domain_vars):
    x_min, x_max = domain_vars.spatial['x']
    y_min, y_max = domain_vars.spatial['y']
    steady_heat_scaling = Scale(domain_vars)
    geom = dde.geometry.Rectangle((steady_heat_scaling.scale_x(x_min), steady_heat_scaling.scale_y(y_min)), (steady_heat_scaling.scale_x(x_max), steady_heat_scaling.scale_y(y_max)))

    bc_left = dde.DirichletBC(geom, lambda x: scale_value(100.0), lambda x, on_boundary: boundary_left(x, on_boundary, x_min, steady_heat_scaling))
    bc_right = dde.DirichletBC(geom, lambda x: scale_value(0.0), lambda x, on_boundary: boundary_right(x, on_boundary, x_max, steady_heat_scaling))

    data = dde.data.PDE(geom, 
                        lambda x,y: steady_lp_residual(x, y, steady_heat_scaling), 
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
    transient_heat_scaling = Scale(domain_vars)
    geom = dde.geometry.Rectangle((transient_heat_scaling.scale_x(x_min), transient_heat_scaling.scale_y(y_min)), (transient_heat_scaling.scale_x(x_max), transient_heat_scaling.scale_y(y_max)))

    timedomain = dde.geometry.TimeDomain(transient_heat_scaling.scale_t(t_min), transient_heat_scaling. scale_t(t_max))   # mit L^2/(pi^2*alpha) geschätzt, und wieder geändert
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    
    # BC
    bc_left = dde.DirichletBC(
        geomtime, 
        lambda x: scale_value(100.0), lambda x, on_boundary: boundary_left(x, on_boundary, x_min, transient_heat_scaling))
    bc_right = dde.DirichletBC(
        geomtime, 
        lambda x: scale_value(0.0), lambda x, on_boundary: boundary_right(x, on_boundary, x_max, transient_heat_scaling))

    # IC
    ic = dde.IC(geomtime, 
        lambda x: scale_value(0.0), lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime,
        lambda x,y: lp_residual(x, y, transient_heat_scaling),
        [bc_left, bc_right, ic],
        num_domain=1600,
        num_boundary=600,
        num_initial=400
    )
    return data
