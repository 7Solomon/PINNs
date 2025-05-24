from utils.metadata import Domain
from process.moisture.scale import *
import deepxde as dde
import numpy as np

from process.moisture.residual import HC, residual

def x_boundary_value(x, on_boundary, values: list[dict]):
    return on_boundary and any([_['intervall'][0]<= x[1] <= _['intervall'][1] and np.isclose(x[0], _['value']) for _ in values])
def y_boundary_value(x, on_boundary, values: list[dict]):
    return on_boundary and any([_['intervall'][0]<= x[0] <= _['intervall'][1] and np.isclose(x[1], _['value']) for _ in values])
def time_boundary_value(x, on_boundary, value:float):
    return on_boundary and np.isclose(x[2], value)

def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], scale_z(0))
def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], scale_z(1))
def boundary_initial(x, on_initial):
    return on_initial and np.isclose(x[1], scale_t(0.0))


def get_simple_domain():
    geom = dde.geometry.geometry_2d.Rectangle((0,0),(1,2))
    time = dde.geometry.TimeDomain(0, scale_t(1e10))   # mit L^2/(K_S/C)
    geomTime = dde.geometry.GeometryXTime(geom, time)

    bc_initial = dde.IC(geomTime, lambda x: -1.0,
                lambda x, on_initial: time_boundary_value(x, on_initial, 0))

    bc_left = dde.DirichletBC(geomTime, lambda x: 0, 
                lambda x, on_boundary: y_boundary_value(x, on_boundary, [
                {'intervall': (0.0,1.0), 'value': 0},
                ]))
    bc_right = dde.DirichletBC(geomTime, lambda x: 0, 
                lambda x, on_boundary: y_boundary_value(x, on_boundary, [
                {'intervall': (0.0,1.0), 'value': 0},
                ]))
    
    bc_bottom = dde.NeumannBC(geomTime, lambda x: -0.01,  # annahme K(u) = 1, also keine gravitaion
                lambda x, on_boundary: x_boundary_value(x, on_boundary, [
                {'intervall': (0,1.0), 'value': 0},
                ]))
    bc_top = dde.DirichletBC(geomTime, lambda x: 0, 
                lambda x, on_boundary: x_boundary_value(x, on_boundary, [ 
                {'intervall': (0,0.5), 'value': 0},
                {'intervall': (0.5,1), 'value': 0.01}  
                ]))

    return dde.data.TimePDE(geomTime,
                        residual,
                        [bc_left, bc_right, bc_bottom, bc_top, bc_initial],
                        num_initial=100,
                        num_domain=200,
                        num_boundary=50
                        
                    )
def get_1d_domain():

    domain = Domain(
        spatial={
            'z': (0, 1),
        }, temporal={
            't': (0, 1.1e10)
        }
    )

    geom = dde.geometry.Interval(0,scale_z(1))
    time = dde.geometry.TimeDomain(0, scale_t(1e10))   # mit L^2/(K_S/C)
    geomTime = dde.geometry.GeometryXTime(geom, time)

    bc_initial = dde.IC(geomTime, lambda x: scale_h(-10),
                boundary_initial)
    bc_left = dde.NeumannBC(geomTime, lambda x: scale_h(-10),  # close to NO FLUX
                boundary_left)
    bc_right = dde.DirichletBC(geomTime, lambda x: scale_h(-9558),   # hs = (R*Tk)/(Mw*g)*ln(RH)   # RH = 0.5  Tk = 293.15
                boundary_right)

    data = dde.data.TimePDE(geomTime,
                        residual,
                        [bc_left, bc_right, bc_initial],
                        num_initial=1000,
                        num_domain=2000,
                        num_boundary=500
                        
                    )
    
    return data, domain
def get_domain(type):
    if type == '2d_head':
        return get_simple_domain()
    elif type == '1d_head':
        return get_1d_domain()