from utils.metadata import Domain
from process.moisture.scale import *
import deepxde as dde
import numpy as np
import torch

#from process.moisture.residual import *
from process.moisture.residualV2 import *

def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], scale_z(0))
def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], scale_z(1))
def boundary_initial(x, on_initial):
    return on_initial and np.isclose(x[1], scale_t(0.0))

def get_r_domain(domain_vars, scale: Scale):
    z_min, z_max = domain_vars.spatial['z']
    t_min, t_max = domain_vars.temporal['t']

    geom = dde.geometry.Interval(z_min/scale.L, z_max/scale.L)
    time = dde.geometry.TimeDomain(t_min/scale.T, t_max/scale.T)
    geomTime = dde.geometry.GeometryXTime(geom, time)


    return geomTime
def get_r_boundary(geomTime, initial_value, left_value, right_value, scale_value=None):
    if scale_value is not None:
        initial_value = scale_value(initial_value)
        left_value = scale_value(left_value)
        right_value = scale_value(right_value)
    
    bc_initial = dde.IC(geomTime, lambda x: initial_value,
                boundary_initial)

    bc_left = dde.DirichletBC(geomTime, lambda x: left_value,
                boundary_left)

    bc_right = dde.DirichletBC(geomTime, lambda x: right_value, 
                boundary_right)
    return  bc_initial, bc_left, bc_right,

def get_1d_mixed_domain(domain_vars):
    geomTime = get_r_domain(domain_vars)
    bc_initial, bc_left, bc_right = get_r_boundary(geomTime, -0.01, -0.01, -200, scale_value=scale_h)
    data = dde.data.TimePDE(geomTime,
                        residual_1d_mixed,
                        [bc_left, bc_right, bc_initial],
                        num_initial=1000,
                        num_domain=2000,
                        num_boundary=300
                    )
    return data
def get_1d_head_domain(domain_vars):

    scale = Scale(domain_vars)

    geomTime = get_r_domain(domain_vars, scale)
    bc_initial, bc_left, bc_right = get_r_boundary(geomTime, -0.01, -0.01, -10, 
                                                   scale_value= lambda x: x / scale.H)

    data = dde.data.TimePDE(geomTime,
                        lambda x,y: residual_1d_head(x,y,scale),
                        [bc_left, bc_right, bc_initial],
                        num_initial=1000,
                        num_domain=2000,
                        num_boundary=300
    )    
    return data
def get_1d_saturation_domain(domain_vars):
    scale = Scale(domain_vars)

    geomTime = get_r_domain(domain_vars, scale)
    bc_initial, bc_left, bc_right = get_r_boundary(geomTime, 0.3, 0.3, 0.01, scale_value=None)

    data = dde.data.TimePDE(geomTime,
                            lambda x,y: residual_1d_saturation(x,y, scale),
                        [bc_left, bc_right, bc_initial],
                        num_initial=1000,
                        num_domain=2000,
                        num_boundary=300
                    )
    return data
def get_d_domain(domain_vars):
    x_min, x_max = domain_vars.spatial['x']
    z_min, z_max = domain_vars.spatial['z']

    geom = dde.geometry.Rectangle(
        (scale_x(x_min), scale_z(z_min)), 
        (scale_x(x_max), scale_z(z_max))
    )

    return geom
def get_d_boundary(geom, left_value, right_value, scale_value=None):
    if scale_value is not None:
        left_value = scale_value(left_value)
        right_value = scale_value(right_value)

    bc_left = dde.DirichletBC(geom, lambda x: left_value,
                boundary_left)

    bc_right = dde.DirichletBC(geom, lambda x: right_value, 
                boundary_right)
    return bc_left, bc_right

def get_2d_darcy_domain(domain_vars):
    geom = get_d_domain(domain_vars)
    bc_left, bc_right = get_d_boundary(geom, -0.01, -200, scale_value=scale_h)

    data = dde.data.PDE(geom,
                        residual_2d_darcy,
                        [bc_left, bc_right],
                        num_domain=2000,
                        num_boundary=300
                    )
    return data


