"""
Domain configuration for moisture transport PINNs.

Simplified and consistent domain creation for different moisture transport problems.
"""

from utils.metadata import Domain
from process.moisture.scale import Scale
import deepxde as dde
import numpy as np
import torch

from process.moisture.residualV2 import *


# =============================================================================
# BOUNDARY CONDITION FUNCTIONS
# =============================================================================

def boundary_left(x, on_boundary):
    """Left boundary condition (x=0)."""
    return on_boundary and np.isclose(x[0], 0)

def boundary_right(x, on_boundary):
    """Right boundary condition (x=1)."""
    return on_boundary and np.isclose(x[0], 1)

def boundary_initial(x, on_initial):
    """Initial time condition (t=0)."""
    return on_initial and np.isclose(x[1], 0)


# =============================================================================
# DOMAIN CREATION
# =============================================================================

def create_1d_time_domain(domain_vars, scale: Scale):
    """Create scaled 1D spatial + time domain."""
    z_min, z_max = domain_vars.spatial['z']
    t_min, t_max = domain_vars.temporal['t']

    geom = dde.geometry.Interval(z_min/scale.L, z_max/scale.L)
    time = dde.geometry.TimeDomain(t_min/scale.T, t_max/scale.T)
    
    return dde.geometry.GeometryXTime(geom, time)


def create_2d_domain(domain_vars, scale: Scale):
    """Create scaled 2D spatial domain."""
    x_min, x_max = domain_vars.spatial['x']
    z_min, z_max = domain_vars.spatial['z']
    
    return dde.geometry.Rectangle(
        (x_min/scale.L, z_min/scale.L), 
        (x_max/scale.L, z_max/scale.L)
    )


def create_boundary_conditions(geom_time, initial_val, left_val, right_val, component=0):
    """Create boundary conditions for 1D time problems."""
    bc_initial = dde.IC(geom_time, lambda x: initial_val, boundary_initial, component=component)
    bc_left = dde.DirichletBC(geom_time, lambda x: left_val, boundary_left, component=component)
    bc_right = dde.DirichletBC(geom_time, lambda x: right_val, boundary_right, component=component)
    
    return [bc_left, bc_right, bc_initial]


def create_2d_boundary_conditions(geom, left_val, right_val):
    """Create boundary conditions for 2D problems."""
    bc_left = dde.DirichletBC(geom, lambda x: left_val, boundary_left)
    bc_right = dde.DirichletBC(geom, lambda x: right_val, boundary_right)
    
    return [bc_left, bc_right]


# =============================================================================
# SPEC
# =============================================================================

def get_1d_head_domain(domain_vars):
    """1D hydraulic head problem with proper scaling."""
    scale = Scale(domain_vars)
    geom_time = create_1d_time_domain(domain_vars, scale)
    
    boundary_conditions = create_boundary_conditions(
        geom_time,
        initial_val=-1.0/scale.H,        #-0.01 / scale.H,  # very high grads which leads to very high diffrence between BC, IC and the resi
        left_val=-0.5/scale.H,        #-0.01 / scale.H,
        right_val=-5.0/scale.H,        #    -10 / scale.H
    )

    return dde.data.TimePDE(
        geom_time,
        lambda x, y: residual_1d_head(x, y, scale),
        boundary_conditions,
        num_initial=1000,
        num_domain=2000,
        num_boundary=300
    )


def get_1d_saturation_domain(domain_vars):
    """1D saturation problem (no scaling needed for saturation)."""
    scale = Scale(domain_vars)
    geom_time = create_1d_time_domain(domain_vars, scale)
    
    boundary_conditions = create_boundary_conditions(
        geom_time,
        initial_val= 0.06/scale.theta,       #0.3,    # Makers very steep gradient with also way to large time scale
        left_val= 0.06/scale.theta,          #0.3,
        right_val=0.3/scale.theta,          #0.01
    )

    return dde.data.TimePDE(
        geom_time,
        lambda x, y: residual_1d_saturation(x, y, scale),
        boundary_conditions,
        num_initial=1000,
        num_domain=2000,
        num_boundary=300
    )




def get_1d_mixed_domain(domain_vars):
    """1D mixed formulation problem."""
    scale = Scale(domain_vars)
    geom_time = create_1d_time_domain(domain_vars, scale)
    
    boundary_conditions = create_boundary_conditions(
        geom_time,
        initial_val=-1.0/scale.H,       
        left_val=-0.5/scale.H,       
        right_val=-5.0/scale.H,
        component=0  # Head
    )

    return dde.data.TimePDE(
        geom_time,
        lambda x, y: residual_1d_mixed(x, y, scale),
        boundary_conditions,
        num_initial=1000,
        num_domain=2000,
        num_boundary=300
    )


def get_2d_darcy_domain(domain_vars):
    """2D Darcy flow problem."""
    scale = Scale(domain_vars)
    geom = create_2d_domain(domain_vars, scale)
    
    boundary_conditions = create_2d_boundary_conditions(
        geom,
        left_val=-0.01, 
        right_val=-200
    )

    return dde.data.PDE(
        geom,
        lambda x, y: residual_2d_darcy(x, y, scale),
        boundary_conditions,
        num_domain=2000,
        num_boundary=300
    )

