import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Callable
from utils import DomainV2, Condition, ConditionType

def create_domain(
    dimensions: Dict[str, Tuple[float, float]],
    n_collocation: int,
    n_boundary: Union[int, Dict[str, int]],
    boundary_conditions: Optional[List[Dict]] = None,
    initial_conditions: Optional[List[Dict]] = None,
    time_dimension: Optional[str] = None
) -> DomainV2:
    """
    Create a domain for PDE solving with flexible dimensionality and conditions.
    
    Args:
        dimensions: Dictionary mapping dimension names to (min, max) ranges
        n_collocation: Number of interior collocation points
        n_boundary: Either a single number for all boundaries or a dict mapping faces to point counts
        boundary_conditions: List of boundary condition specifications
        initial_conditions: List of initial condition specifications (for time-dependent problems)
        time_dimension: Name of time dimension (if any)
    
    Returns:
        Domain object with all necessary points and conditions
    """
    domain = DomainV2(dimensions=dimensions, time_dim=time_dimension)
    domain.is_time_dependent = time_dimension is not None
    
    # Generate interior collocation points
    domain.collocation_points = generate_collocation_points(dimensions, n_collocation)
    
    # Generate boundary points for each face
    domain.boundary_points = generate_boundary_points(dimensions, n_boundary)
    
    # Process boundary conditions
    if boundary_conditions:
        for bc in boundary_conditions:
            condition = Condition(
                type=bc['type'],
                value=bc['value'],
                location=bc.get('face'),
                params=bc.get('params', {})
            )
            domain.boundary_conditions.append(condition)
    
    # Process initial conditions for time-dependent problems
    if domain.is_time_dependent and initial_conditions:
        for ic in initial_conditions:
            condition = Condition(
                type=ConditionType.INITIAL,
                value=ic['value'],
                params=ic.get('params', {})
            )
            domain.initial_conditions.append(condition)
            
        # Generate initial condition points if needed
        domain.initial_points = generate_initial_points(dimensions, time_dimension)
    
    return domain

def generate_collocation_points(dimensions: Dict[str, Tuple[float, float]], n_points: int) -> torch.Tensor:
    """Generate random points inside the domain"""
    dim_names = list(dimensions.keys())
    dim_ranges = list(dimensions.values())
    ndims = len(dim_names)
    
    points = torch.zeros((n_points, ndims))
    for i, (min_val, max_val) in enumerate(dim_ranges):
        points[:, i] = min_val + (max_val - min_val) * torch.rand(n_points)
    
    return points

def generate_boundary_points(
    dimensions: Dict[str, Tuple[float, float]], 
    n_boundary: Union[int, Dict[str, int]]
) -> Dict[str, torch.Tensor]:
    """Generate points on each boundary face"""
    boundary_points = {}
    dim_names = list(dimensions.keys())
    ndims = len(dim_names)
    
    # Handle both uniform and per-face point counts
    if isinstance(n_boundary, int):
        points_per_face = {f"{dim}_{side}": n_boundary 
                          for dim in dim_names 
                          for side in ["min", "max"]}
    else:
        points_per_face = n_boundary
    
    # Generate points for each face
    for dim_idx, dim_name in enumerate(dim_names):
        for side in ["min", "max"]:
            face_name = f"{dim_name}_{side}"
            n_points = points_per_face.get(face_name, 0)
            
            if n_points > 0:
                face_points = torch.rand(n_points, ndims)
                # Set the boundary dimension to its min/max value
                face_points[:, dim_idx] = dimensions[dim_name][0 if side == "min" else 1]
                
                # Create random points for other dimensions
                for j, other_dim in enumerate(dim_names):
                    if j != dim_idx:
                        min_val, max_val = dimensions[other_dim]
                        face_points[:, j] = min_val + (max_val - min_val) * face_points[:, j]
                
                boundary_points[face_name] = face_points
    
    return boundary_points

def generate_initial_points(
    dimensions: Dict[str, Tuple[float, float]],
    time_dimension: str,
) -> torch.Tensor:
    """Generate points at the initial time for time-dependent problems"""
    # Filter out time dimension to get spatial dimensions only
    spatial_dims = {k: v for k, v in dimensions.items() if k != time_dimension}
    
    # Number of points should be sufficient to cover the domain
    n_points = 1000  # This could be made configurable
    
    # Generate points for spatial dimensions
    points = generate_collocation_points(spatial_dims, n_points)
    
    # Add time dimension (set to minimum time value)
    time_idx = list(dimensions.keys()).index(time_dimension)
    time_min = dimensions[time_dimension][0]
    
    # Create full points including time
    full_points = torch.zeros((n_points, len(dimensions)))
    
    # Copy spatial dimensions
    spatial_idx = 0
    for i in range(full_points.shape[1]):
        if i == time_idx:
            full_points[:, i] = time_min
        else:
            full_points[:, i] = points[:, spatial_idx]
            spatial_idx += 1
    
    return full_points