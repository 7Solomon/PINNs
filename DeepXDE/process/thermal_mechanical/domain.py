
import torch
import deepxde as dde

from process.thermal_mechanical.residual import residual_thermal_2d
from process.thermal_mechanical.scale import Scale

def get_thermal_2d_domain(domain_vars):
    scale = Scale(domain_vars)
    x_min, x_max = domain_vars.spatial['x']
    y_min, y_max = domain_vars.spatial['y']

    t_min, t_max = domain_vars.temporal['t']

    geom = dde.geometry.Rectangle(
        (x_min/scale.Lx, y_min/scale.Ly), 
        (x_max/scale.Lx, y_max/scale.Ly)
    )
    time = dde.geometry.TimeDomain(t_min/scale.t, t_max/scale.t)
    geomTime = dde.geometry.GeometryXTime(geom, time)

    value_left = 100.0 / scale.Temperature
    value_right = 0.0 / scale.Temperature
    initial_value = 0.0 / scale.Temperature

    left_boundary = dde.DirichletBC(
        geomTime, 
        lambda x: value_left, 
        lambda x, on_boundary: on_boundary and x[0] == x_min,
        component=2
    )
    right_boundary = dde.DirichletBC(
        geomTime, 
        lambda x: value_right, 
        lambda x, on_boundary: on_boundary and x[0] == x_max,
        component=2
    )
    initial_condition = dde.IC(
        geomTime, 
        lambda x: initial_value, 
        lambda _, on_initial: on_initial
    )

    data = dde.data.TimePDE(
        geomTime,
        lambda x,y : residual_thermal_2d(x,y, scale),
        [left_boundary, right_boundary, initial_condition],
        num_initial=100,
        num_domain=200,
        num_boundary=30
    )
    return data
