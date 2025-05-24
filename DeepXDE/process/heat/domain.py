from utils.metadata import Domain
from process.heat.scale import *
import numpy as np
import deepxde as dde
from process.heat.residual import lp_residual, steady_lp_residual


def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], scale_x(0))
def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], scale_x(2))

def get_steady_domain():
    domain = Domain(spatial={
        'x': (0, 2),
        'y': (0, 1)
    }, temporal=None)

    geom = dde.geometry.Rectangle((0,0),(scale_x(2),scale_y(1)))

    bc_left = dde.DirichletBC(geom, lambda x: scale_value(100.0), boundary_left)
    bc_right = dde.DirichletBC(geom, lambda x: scale_value(0.0), boundary_right)
    
    data = dde.data.PDE(geom, 
                        steady_lp_residual, 
                        [bc_left, bc_right], 
                        num_domain=200, 
                        num_boundary=50)
    # wights 
    #data.set_weights([1, 1, 1, 1])
    
    return data

def get_transient_domain():
    domain = Domain(spatial={
        'x': (0, 2),
        'y': (0, 1)
    }, temporal={
        't': (0, 1.1e7)
    })
    geom = dde.geometry.Rectangle((0, 0), (scale_x(2), scale_y(1)))

    timedomain = dde.geometry.TimeDomain(0, scale_time(1.1e7))   # mit L^2/(pi^2*alpha) geschätzt
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    
    # BC
    bc_left = dde.DirichletBC(
        geomtime, 
        lambda x: scale_value(100.0), boundary_left)
    bc_right = dde.DirichletBC(
        geomtime, 
        lambda x: scale_value(0.0), boundary_right)
    
    # IC
    ic = dde.IC(geomtime, 
        lambda x: scale_value(0.0), lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime,
        lp_residual,
        [bc_left, bc_right, ic],
        num_domain=200,
        num_boundary=50,
        num_initial=100
    )
    return data, domain

def get_domain(type):
    if type == 'steady':
        return get_steady_domain()
    elif type == 'transient':
        return get_transient_domain()
    else:
        raise ValueError(f'Unbekannter Typ: {type}')