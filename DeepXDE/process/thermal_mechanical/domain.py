
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
        (x_min/scale.L, y_min/scale.L), 
        (x_max/scale.L, y_max/scale.L)
    )
    time = dde.geometry.TimeDomain(t_min/scale.t, t_max/scale.t)
    geomTime = dde.geometry.GeometryXTime(geom, time)

    value_left = 1.0 / scale.Temperature
    value_right = 0.0 / scale.Temperature
    initial_value = 0.5 / scale.Temperature

    left_temp_boundary = dde.DirichletBC(
        geomTime, 
        lambda x: value_left, 
        lambda x, on_boundary: on_boundary and x[0] == (x_min/scale.L),
        component=2
    )
    right_temp_boundary = dde.DirichletBC(
        geomTime, 
        lambda x: value_right, 
        lambda x, on_boundary: on_boundary and x[0] == (x_max/scale.L),
        component=2
    )
    left_u_fixed = dde.DirichletBC(
        geomTime,
        lambda x: 0.0,
        lambda x, on_boundary: on_boundary and x[0] == (x_min/scale.L),
        component=0
    )
    bottom_v_fixed = dde.DirichletBC(
        geomTime,
        lambda x: 0.0,
        lambda x, on_boundary: on_boundary and x[1] == (y_min/scale.L),
        component=1
    )

    initial_condition = dde.IC(
        geomTime, 
        lambda x: initial_value, 
        lambda _, on_initial: on_initial
    )

    data = dde.data.TimePDE(
        geomTime,
        lambda x,y : residual_thermal_2d(x,y, scale),
        [left_temp_boundary, right_temp_boundary, left_u_fixed, bottom_v_fixed, initial_condition],
        num_initial=1000,
        num_domain=2000,
        num_boundary=300
    )
    return data
