import numpy as np
import deepxde as dde
from process.mechanic.residual import pde_residual

def scale_value(T):
    return T/100


def du_dxx(x, y, _):
    return dde.grad.hessian(y, x)[:,0]
def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)
def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)
def get_fest_los_domain():
    geom = dde.geometry.Interval(0,1)

    bc_left_w = dde.DirichletBC(geom, lambda x: 0, boundary_left)
    bc_right_w = dde.DirichletBC(geom, lambda x: 0, boundary_right)

    bc_left_wxx = dde.OperatorBC(geom, du_dxx, boundary_left)
    bc_right_wxx = dde.OperatorBC(geom, du_dxx, boundary_right)
    #bc_left_w = dde.icbc.boundary_conditions.DirichletBC(geom, lambda x: 0, boundary_left)
    #bc_right_w = dde.icbc.boundary_conditions.DirichletBC(geom, lambda x: 0, boundary_right)
    #bc_left_wxx = dde.icbc.boundary_conditions.OperatorBC(geom, du_dxx, boundary_left)
    #bc_right_wxx = dde.icbc.boundary_conditions.OperatorBC(geom, du_dxx, boundary_right)
    
    
    
    data = dde.data.PDE(geom, 
                        pde_residual, 
                        [bc_left_w, bc_right_w, bc_left_wxx, bc_right_wxx], 
                        num_domain=200, 
                        num_boundary=50)
    # wights 
    #data.set_weights([1, 1, 1, 1])
    return data

def get_domain(type):
    if type == 'fest_los':
        return get_fest_los_domain()
    elif type == 'einspannung':
        raise NotImplementedError('Einsparung is not implemented yet')
    else:
        raise NotImplementedError(f'Domain {type} is not implemented yet')