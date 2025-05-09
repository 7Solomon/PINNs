import numpy as np
import deepxde as dde
from residual import pde_residual

def du_dx(x, y, X):
    return dde.grad.jacobian(y, x, i=0)
def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)
def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)
def get_domain():
    geom = dde.geometry.Interval(0,1)

    # Displacement = 0 at boundaries
    bc_left_w = dde.DirichletBC(geom, lambda x: 0, boundary_left)
    bc_right_w = dde.DirichletBC(geom, lambda x: 0, boundary_right)
    
    # Slope = 0 at boundaries (using OperatorBC instead of NeumannBC)
    bc_left_wx = dde.OperatorBC(geom, du_dx, boundary_left)
    bc_right_wx = dde.OperatorBC(geom, du_dx, boundary_right)
    
    
    data = dde.data.PDE(geom, 
                        pde_residual, 
                        [bc_left_w, bc_right_w, bc_left_wx, bc_right_wx], 
                        num_domain=200, 
                        num_boundary=50)
    # wights 
    #data.set_weights([1, 1, 1, 1])
    return data