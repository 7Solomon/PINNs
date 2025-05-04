import torch
import numpy as np

from utils import Domain, Condition, ConditionType


def get_collocation(n_points=1000, x=[0,1], y=[0,2], t= [0,10]):
    x = torch.rand(n_points, 1, requires_grad=True) * (x[1] - x[0]) + x[0]
    y = torch.rand(n_points, 1, requires_grad=True) * (y[1] - y[0]) + y[0]
    t = torch.rand(n_points, 1, requires_grad=True) * (t[1] - t[0]) + t[0]
    return torch.cat([x, y, t], dim=1)  # [n,3] tensor

def get_initial_conditions(n_points=200, x=[0,1], y=[0,2], h_0=0.0):
    x = torch.rand(n_points, 1) * (x[1] - x[0]) + x[0]
    y = torch.rand(n_points, 1) * (y[1] - y[0]) + y[0]
    t = torch.full((n_points, 1), 0)

    XYT = torch.cat([x, y, t], dim=1)
    values = torch.full_like(x, h_0)
    return Condition(
                        key='initial',
                        type=ConditionType.INITIAL, 
                        points=XYT,
                        values=values,
                    )
        

def get_boundary_condition(n_points=200, x=[0,1], y=[0,2], t= [0,10]):
    t = torch.rand(n_points, 1, requires_grad=True) * (t[1] - t[0]) + t[0]
    x_left = torch.full((n_points, 1), x[0])
    y_left = torch.rand(n_points,1, requires_grad=True) * (y[1] - y[0]) + y[0]
    bc_left = torch.cat([x_left,y_left,t], dim=1)
    bc_left_values = torch.full_like(x_left, 0.0, dtype=torch.float32)   # !!! anders lösen
    left = Condition(
        key='left',
        type=ConditionType.DIRICHTLETT,
        points=bc_left,
        values=bc_left_values,
    )

    x_right = torch.full((n_points, 1), x[1])
    y_right = torch.rand(n_points,1, requires_grad=True) * (y[1] - y[0]) + y[0]
    bc_right = torch.cat([x_right,y_right,t], dim=1)
    bc_right_values = torch.full_like(x_right, 0.0, dtype=torch.float32)  ## !!!
    right = Condition(
        key='right',
        type=ConditionType.DIRICHTLETT,
        points=bc_right,
        values=bc_right_values,
    )
    
    x_top = torch.rand(n_points, 1, requires_grad=True) * (x[1] - x[0]) + x[0]
    y_top = torch.full((n_points, 1), y[1])
    bc_top = torch.cat([x_top,y_top,t], dim=1)
    bc_top_values = torch.full_like(x_top,  -30.0, dtype=torch.float32) ##!!
    top = Condition(
        key='top',
        type=ConditionType.DIRICHTLETT,
        points=bc_top,
        values=bc_top_values,
    )

    x_bottom = torch.rand(n_points, 1, requires_grad=True) * (x[1] - x[0]) + x[0]
    y_bottom = torch.full((n_points, 1), y[0])
    bc_bottom = torch.cat([x_bottom,y_bottom,t], dim=1)
    bc_bottom_values = torch.full_like(x_bottom, 0.0, dtype=torch.float32) ##!!
    bottom = Condition(
        key='bottom',
        type=ConditionType.DIRICHTLETT,
        points=bc_bottom,
        values=bc_bottom_values,
    )
    return {
            'left': left,
            'right': right,
            'top': top,
            'bottom': bottom,
        }

def get_domain():
    domain = Domain()
    domain.header = {'x': (0, 1), 'y': (0, 2), 't': (0, 10)}
    domain.collocation = get_collocation()
    domain.initial_condition = get_initial_conditions()
    boundary_conditions = get_boundary_condition()
    domain.condition_keys = list(boundary_conditions.keys())
    domain.conditions = boundary_conditions
    return domain

    
