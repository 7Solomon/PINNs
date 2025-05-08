import torch
import numpy as np

from utils import Domain, Condition, ConditionType


def get_collocation(n_points=1000, z=[0,2], t= [0,10]):
    z_coords = torch.rand(n_points, 1, requires_grad=True) * (z[1] - z[0]) + z[0]
    t_coords = torch.rand(n_points, 1, requires_grad=True) * (t[1] - t[0]) + t[0]
    return torch.cat([z_coords, t_coords], dim=1)  # [n,2] tensor

def get_initial_conditions(n_points=200, z=[0,2], psi=0.0):
    z_coords = torch.rand(n_points, 1) * (z[1] - z[0]) + z[0]
    t_coords = torch.full((n_points, 1), 0)

    ZT = torch.cat([z_coords, t_coords], dim=1)
    values = torch.full_like(z_coords, psi)
    return Condition(
                        key='initial',
                        type=ConditionType.INITIAL,
                        points=ZT,
                        values=values,
                    )


# Updated for z boundaries (z_min, z_max)
def get_boundary_condition(n_points=200, z=[0,2], t= [0,10]):
    t_coords = torch.rand(n_points, 1, requires_grad=True) * (t[1] - t[0]) + t[0]

    # Boundary at z = z[0] (e.g., z=0)
    z_min_coords = torch.full((n_points, 1), z[0])
    bc_z_min_points = torch.cat([z_min_coords, t_coords], dim=1)
    bc_z_min_values = torch.full_like(z_min_coords, -30.0, dtype=torch.float32) # Example value
    z_min_condition = Condition(
        key='z_min',
        type=ConditionType.DIRICHLET,
        points=bc_z_min_points,
        values=bc_z_min_values,
    )

    # Boundary at z = z[1] (e.g., z=2)
    z_max_coords = torch.full((n_points, 1), z[1])
    bc_z_max_points = torch.cat([z_max_coords, t_coords], dim=1)
    bc_z_max_values = torch.full_like(z_max_coords, 0, dtype=torch.float32) # Example value
    z_max_condition = Condition(
        key='z_max',
        type=ConditionType.DIRICHLET,
        points=bc_z_max_points,
        values=bc_z_max_values,
    )

    return {
            'z_min': z_min_condition,
            'z_max': z_max_condition,
        }

# Updated to use z, t
def get_domain(conf):
    domain = Domain()
    domain.header = {'z': conf.z_range, 't': conf.t_range}
    domain.collocation = get_collocation(z=conf.z_range, t=conf.t_range)
    domain.initial_condition = get_initial_conditions(z=conf.z_range)
    boundary_conditions = get_boundary_condition(z=conf.z_range, t=conf.t_range)
    domain.condition_keys = list(boundary_conditions.keys()) 
    domain.conditions = boundary_conditions
    return domain