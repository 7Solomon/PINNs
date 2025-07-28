
from utils.metadata import Domain
import torch
import numpy as np
import deepxde as dde

from process.thermal_mechanical.residual import residual_thermal_2d
from process.thermal_mechanical.scale import Scale

def get_thermal_2d_domain(domain_vars: Domain, scale: Scale):
    x_min, x_max = domain_vars.spatial['x']
    y_min, y_max = domain_vars.spatial['y']

    t_min, t_max = domain_vars.temporal['t']

    geom = dde.geometry.Rectangle(
        (x_min/scale.L, y_min/scale.L), 
        (x_max/scale.L, y_max/scale.L)
    )
    time = dde.geometry.TimeDomain(t_min/scale.t, t_max/scale.t)
    geomTime = dde.geometry.GeometryXTime(geom, time)

    value_left = 50.0 / scale.Temperature
    value_right = 0.0 / scale.Temperature
    initial_value = 10.0 / scale.Temperature

    left_temp_boundary = dde.DirichletBC(
        geomTime, 
        lambda x: value_left, 
        lambda x, on_boundary: on_boundary and np.isclose(x[0], (x_min/scale.L)),
        component=2
    )
    right_temp_boundary = dde.DirichletBC(
        geomTime, 
        lambda x: value_right, 
        lambda x, on_boundary: on_boundary and np.isclose(x[0], (x_max/scale.L)),
        component=2
    )
    
    bottom_u_fixed = dde.DirichletBC(
        geomTime,
        lambda x: 0.0,
        lambda x, on_boundary: on_boundary and np.isclose(x[1], (y_min/scale.L)),
        component=0
    )
    bottom_v_fixed = dde.DirichletBC(
        geomTime,
        lambda x: 0.0,
        lambda x, on_boundary: on_boundary and np.isclose(x[1], (y_min/scale.L)),
        component=1
    )

    initial_temp = dde.IC(
        geomTime, 
        lambda x: initial_value, 
        lambda _, on_initial: on_initial,
        component=2
    )

    initial_u = dde.IC(
        geomTime,
        lambda x: 0.0,
        lambda _, on_initial: on_initial,
        component=0
    )

    initial_v = dde.IC(
        geomTime,
        lambda x: 0.0,
        lambda _, on_initial: on_initial,
        component=1
    )

    data = dde.data.TimePDE(
        geomTime,
        lambda x,y : residual_thermal_2d(x,y, scale),
        #[left_temp_boundary, right_temp_boundary, bottom_u_fixed, bottom_v_fixed, initial_temp, initial_u, initial_v],
        [left_temp_boundary, right_temp_boundary, bottom_u_fixed, bottom_v_fixed, initial_temp],
        num_initial=1000,
        num_domain=2000,
        num_boundary=300
    )
    return data
