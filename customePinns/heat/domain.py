from utils import Condition, ConditionType, Domain

from heat.vars import *
import numpy as np
import torch

def generate_random_collocation(domain: Domain, n_points, x_min=0.0, x_max=2.0, y_min=0.0, y_max=1.0):  # maybe time add ?
    """
    Das hier ist für Steady State, also keine Zeit.
    Returns:
        numpy.ndarray: [n_points, 2] => (x, y).
    """
    x_collocation = np.random.rand(n_points, 1) * (x_max - x_min) + x_min
    y_collocation = np.random.rand(n_points, 1) * (y_max - y_min) + y_min
    collocation_points = np.hstack((x_collocation, y_collocation))
    collocation_tensor= torch.tensor(collocation_points, dtype=torch.float32, requires_grad=True)
    domain.collocation = collocation_tensor
    return domain

def generate_random_boundary(domain: Domain , n_points, x_min=0.0, x_max=2.0, y_min=0.0, y_max=1.0):
    """
    Auch gerade nur Steady State, aber boundary bleibt gleivh über zeit
    """
    x_coords_left = np.ones((n_points, 1)) * x_min
    y_coords_left = np.random.rand(n_points, 1) * (y_max - y_min) + y_min
    boundary_points_left = np.hstack((x_coords_left, y_coords_left)) # [500,2]
    boundary_points_left_tensor = torch.tensor(boundary_points_left, dtype=torch.float32, requires_grad=True)

    x_coords_right = np.ones((n_points, 1)) * x_max
    y_coords_right = np.random.rand(n_points, 1) * (y_max - y_min) + y_min
    boundary_points_right = np.hstack((x_coords_right, y_coords_right)) # [500,2]
    boundary_points_right_tensor = torch.tensor(boundary_points_right, dtype=torch.float32, requires_grad=True)
    
    temp_values_left = torch.full((n_points, 1), T_max)# [500,1] = 100
    temp_values_right = torch.full((n_points, 1), T_min)

    domain.header = {'x':(x_min, x_max), 'y':(y_min, y_max)}
    domain.condition_keys = ['left', 'right']

    domain.conditions = {
        'left': 
            Condition(
                        key='left',
                        type=ConditionType.DIRICHTLETT,
                        points=boundary_points_left_tensor,
                        values=temp_values_left,
                    ), 
        'right': 
            Condition(
                        key='right',
                        type=ConditionType.DIRICHTLETT,
                        points=boundary_points_right_tensor,
                        values=temp_values_right,
                    ),
        }
    return domain




def generate_steady_domain():
    num_collocation_points = 5000
    num_boundary_points = 1000
    
    domain = Domain()
    domain = generate_random_collocation(domain, num_collocation_points)
    domain = generate_random_boundary(domain, num_boundary_points)
    #domain = {**boundary, **{'domain_collocation_tensor': domain_collocation_tensor}}
    return domain
