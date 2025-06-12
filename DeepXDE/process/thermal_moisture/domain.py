import deepxde as dde
import numpy as np
from process.thermal_moisture.residual import residual
from process.thermal_moisture.scale import Scale


def get_2d_domain(domain_vars):
    
    scale = Scale(domain_vars)
    x_min, x_max = domain_vars.spatial['x']
    y_min, y_max = domain_vars.spatial['y']

    t_min, t_max = domain_vars.temporal['t']

    geom = dde.geometry.Rectangle(
        (x_min / scale.L, y_min / scale.L), 
        (x_max / scale.L, y_max / scale.L)
    )
    time = dde.geometry.TimeDomain(t_min / scale.t, t_max / scale.t)
    geom_time = dde.geometry.GeometryXTime(geom, time)

    temperature_initial_value = 10.0 / scale.Temperature 
    moisture_initial_value = 0.2 / scale.theta  

    temperature_left_value = 30 / scale.Temperature
    moisture_left_value = 0.005 / scale.theta  


    temperature_initial = dde.IC(
        geom_time,
        lambda x: temperature_initial_value,
        lambda _, on_initial: on_initial,
        component=0
    )
    moisture_initial = dde.IC(
        geom_time,
        lambda x: moisture_initial_value,
        lambda _, on_initial: on_initial,
        component=1
    )
    temperature_left_boundary = dde.DirichletBC(
        geom_time, 
        lambda x: temperature_left_value, 
        lambda x, on_boundary: on_boundary and np.isclose(x[0], (x_min/scale.L)),
        component=0
    )
    moisture_left_boundary = dde.DirichletBC(
        geom_time, 
        lambda x: moisture_left_value, 
        lambda x, on_boundary: on_boundary and np.isclose(x[0], (x_min/scale.L)),
        component=1
    )

    data = dde.data.TimePDE(
        geom_time,
        lambda x, y: residual(x, y, scale),
        [temperature_initial, moisture_initial, temperature_left_boundary, moisture_left_boundary],
        num_domain=1000,
        num_boundary=100,
        num_initial=100,
    )
    return data
