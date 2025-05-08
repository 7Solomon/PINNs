from dataclasses import dataclass, field
from typing import Dict
from utils import ConditionType

import torch

#@dataclass
#class SimpleTimeTestDomain:
#    dimension: Dict[str, tuple[float, float]] = None
#    inputs: Dict[str, tuple[float, float]] = None
#    time_frame: tuple[float, float] = None
#    collocation: torch.Tensor = None
#    boundarys: dict[str, torch.Tensor] = None
#    initials: dict[str, torch.Tensor] = None


def scale_tensor(tensor: torch.Tensor, scale: tuple[float, float]) -> torch.Tensor:
    min_val, max_val = scale
    if tensor.max() == tensor.min():
        mid_point = (min_val + max_val) / 2
        return torch.ones_like(tensor) * mid_point
    return (tensor - tensor.min()) / (tensor.max() - tensor.min()) * (max_val - min_val) + min_val

@dataclass 
class Condition:
    key: str = None
    type: ConditionType = None
    points: torch.Tensor = None
    values: torch.Tensor = None

    def populate(self, value: float) -> None:
        self.values = torch.ones((self.points.shape[0], 1)) * value



@dataclass
class SimpleTestDomain:
    dimension: Dict[str, tuple[float, float]] = field(default_factory=dict)
    inputs: Dict[str, tuple[float, float]] = field(default_factory=dict)
    collocation: torch.Tensor = None
    boundarys: dict[str, Condition] = field(default_factory=dict)

    scale_domains: dict[str, dict[str, tuple[float, float]]] = field(default_factory=lambda:{
        'collocation':{},
        'boundary':{},
    })
    #point_scale_domain: tuple[float,float] = [None,None]
    #value_scale_domain: tuple[float,float] = [None,None]
    



def get_collocation_points(domain: SimpleTestDomain, n_points: int) -> torch.Tensor:
    dim_ranges = list(domain.dimension.values())
    collocation_points = torch.zeros((n_points, len(dim_ranges)))
    for i, (min_val, max_val) in enumerate(dim_ranges):
        collocation_points[:, i] = min_val + (max_val - min_val) * torch.rand(n_points)
    return collocation_points
def get_boundary_points(domain: SimpleTestDomain, n_points: int) -> dict[str, dict[str, torch.Tensor]]:
    boundary_points = {}
    dim_names = list(domain.dimension.keys())
    
    for i, dim_name in enumerate(dim_names):
        min_val, max_val = domain.dimension[dim_name]
        # Min
        min_boundary = torch.zeros((n_points, len(dim_names)))
        for j, other_dim in enumerate(dim_names):
            if j == i:
                min_boundary[:, j] = min_val
            else:
                other_min, other_max = domain.dimension[other_dim]
                min_boundary[:, j] = other_min + (other_max - other_min) * torch.rand(n_points)
        # Max
        max_boundary = torch.zeros((n_points, len(dim_names)))
        for j, other_dim in enumerate(dim_names):
            if j == i:
                max_boundary[:, j] = max_val
            else:
                other_min, other_max = domain.dimension[other_dim]
                max_boundary[:, j] = other_min + (other_max - other_min) * torch.rand(n_points)
        
        boundary_points[dim_name] = {
            'min': min_boundary,
            'max': max_boundary
        }
    return boundary_points

def populate_boundary(domain: SimpleTestDomain, value: float, n_points: int) -> None:
    boundary_points = get_boundary_points(domain, n_points)
    for dim_name, points in boundary_points.items():
        for side, point_set in points.items():
            # HERE ADD SCALING Stuff
            condition = Condition(
                key=f'{dim_name}_{side}',
                type=ConditionType.DIRICHLET,
                points=point_set,
                values=torch.ones((point_set.shape[0], 1)) * value
            )
            domain.boundarys[condition.key] = condition

def populate_collocation_with_constant(domain: SimpleTestDomain, value: float , n_points: int) -> None:
    collocation_points = get_collocation_points(domain, n_points)
    constant = torch.ones((collocation_points.shape[0], 1)) * value

    domain.scale_domains['collocation']['points'] = (collocation_points.min().item(), collocation_points.max().item())
    domain.scale_domains['collocation']['inputs'] = (constant.min().item(), constant.max().item())

    scalled_points = scale_tensor(collocation_points, domain.scale_domains['collocation']['points'])
    scalled_constant= scale_tensor(constant, domain.scale_domains['collocation']['inputs'])

    domain.collocation  = torch.cat((scalled_points, scalled_constant), dim=1).requires_grad_(True)
    #print(f'collocation shape: {domain.collocation.shape}')
    #print(f'collocation: {domain.collocation[:2]}')

    

def get_domain(conf):
    domain = SimpleTestDomain()
    domain.dimension = { 'x': (0.0, 2.0)}
    #domain.inputs = {'q': (0.0, 10.0)}

    populate_collocation_with_constant(domain, 10.0, 1000)
    populate_boundary(domain, 0, 1000)
    return domain


