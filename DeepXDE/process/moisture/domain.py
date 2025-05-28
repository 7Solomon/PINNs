from utils.metadata import Domain
from process.moisture.scale import *
import deepxde as dde
import numpy as np
import torch

from process.moisture.residual import HC, residual_1d_mixed

def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], scale_z(0))
def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], scale_z(1))
def boundary_initial(x, on_initial):
    return on_initial and np.isclose(x[1], scale_t(0.0))

def get_1d_mixed_domain(domain_vars):
    geom = dde.geometry.Interval(0,scale_z(1))
    time = dde.geometry.TimeDomain(0, scale_t(1e10))   # mit L^2/(K_S/C)
    geomTime = dde.geometry.GeometryXTime(geom, time)

    bc_initial = dde.IC(geomTime, lambda x: scale_h(-0.01),
                boundary_initial)
    #bc_left = dde.NeumannBC(geomTime, lambda x: -0.0001, #scale_h(-10),  # close to NO FLUX  # change to NO FLUX
    #            boundary_left)
    bc_left = dde.DirichletBC(geomTime, lambda x: scale_h(-0.01),   # hs = (R*Tk)/(Mw*g)*ln(RH)   # RH = 0.5  Tk = 293.15
                boundary_left)


    bc_right = dde.DirichletBC(geomTime, lambda x: scale_h(-200),   # hs = (R*Tk)/(Mw*g)*ln(RH)   # RH = 0.5  Tk = 293.15
                boundary_right)

    data = dde.data.TimePDE(geomTime,
                        residual_1d_mixed,
                        [bc_left, bc_right, bc_initial],
                        num_initial=1000,
                        num_domain=2000,
                        num_boundary=500
                        
                    )
    
    return data


