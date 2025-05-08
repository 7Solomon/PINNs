from domain.generate import create_domain
import torch
from domain.domain import ConditionType

def get_steady_heat_domain(conf):
    # Define dimensions - easily adaptable to 1D, 2D, or 3D
    dims = {
        'x': (0.0, 1.0),
        'y': (0.0, 2.0),
        # 'z': (0.0, 1.0),  # Uncomment for 3D
    }

    # Define boundary conditions with improved structure
    bcs = [
        {'face': 'x_min', 'type': ConditionType.DIRICHLET, 'value': conf.T_max}, 
        {'face': 'x_max', 'type': ConditionType.DIRICHLET, 'value': conf.T_min},
        {'face': 'y_min', 'type': ConditionType.DIRICHLET, 'value': 0.0},
        {'face': 'y_max', 'type': ConditionType.DIRICHLET, 'value': 0.0} 
    ]

    # Create domain
    domain = create_domain(
        dimensions=dims,
        n_collocation=1000,
        n_boundary=200,  # Use dict for non-uniform point distribution
        boundary_conditions=bcs,
    )
    
    return domain

def get_transient_heat_domain(conf):
    # Time-dependent version
    dims = {
        'x': (0.0, 1.0),
        'y': (0.0, 2.0),
        't': (0.0, 5.0)  # Add time dimension
    }

    bcs = [
        {'face': 'x_min', 'type': ConditionType.DIRICHLET, 'value': conf.T_max}, 
        {'face': 'x_max', 'type': ConditionType.DIRICHLET, 'value': conf.T_min},
        {'face': 'y_min', 'type': ConditionType.DIRICHLET, 'value': 0.0},
        {'face': 'y_max', 'type': ConditionType.DIRICHLET, 'value': 0.0} 
    ]

    ics = [
        {'type': ConditionType.INITIAL, 
         'value': lambda pts: torch.exp(-(pts[:, 0:1]-0.5)**2 - (pts[:, 1:2]-1.0)**2)}
    ]

    domain = create_domain(
        dimensions=dims,
        n_collocation=2000,
        n_boundary=300,
        boundary_conditions=bcs,
        initial_conditions=ics,
        time_dimension='t'
    )
    
    return domain