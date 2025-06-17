import deepxde as dde
import numpy as np
import functools
class ProblemDefinition:
    def __init__(self, name, description, domain_coords, variables, residual_func, scale,
                 boundary_conditions, initial_conditions=None):
        self.name = name
        self.description = description
        self.domain_coords = domain_coords
        self.variables = variables

        self.residual_func = residual_func
        self.scale = scale
        
        self.boundary_conditions = boundary_conditions
        self.initial_conditions = initial_conditions if initial_conditions is not None else []

        # Helper maps
        self.coord_map = {name: i for i, name in enumerate(domain_coords.keys())}
        self.var_map = {name: i for i, name in enumerate(variables)}

    def get_coord_index(self, name):
        return self.coord_map[name]

    def get_variable_index(self, name):
        return self.var_map[name]
    
#######
# Create DDE Conditions
#######


def create_dde_bcs(problem: ProblemDefinition, geom):
    """Generates deepxde BCs from a ProblemDefinition."""
    dde_bcs = []
    def make_dde_selector(location_dict):
        def selector(x, on_boundary):
            # We only care about points on the boundary for BCs. ICs are on the t-boundary.
            if not on_boundary:
                return False
            
            checks = []
            for coord_name, location in location_dict.items():
                coord_idx = problem.get_coord_index(coord_name)
                domain = problem.domain_coords[coord_name]
                
                if location == 'min':
                    val = domain[0]
                elif location == 'max':
                    val = domain[1]
                else: # Assumes a numeric value
                    val = location
                
                checks.append(np.isclose(x[coord_idx], val))

            # Return True only if all checks pass
            return functools.reduce(np.logical_and, checks)
        return selector

    for cond in problem.boundary_conditions:
        selector_func = make_dde_selector(cond['location_on'])
        value_func = cond['value'] if callable(cond['value']) else lambda x: cond['value']
        var_component = problem.get_variable_index(cond['variable'])
        
        if cond['type'] == 'Dirichlet':
            dde_cond = dde.DirichletBC(geom, value_func, selector_func, component=var_component)
        elif cond['type'] == 'Neumann':
            dde_cond = dde.NeumannBC(geom, value_func, selector_func, component=var_component)
        else:
            raise ValueError(f"Unknown condition type: {cond['type']}")

        dde_bcs.append(dde_cond)

    return dde_bcs

def create_dde_ics(problem: ProblemDefinition, geom):
    """Generates a list of dde.ic.IC objects."""
    dde_ic_list = []
    for ic_cond in problem.initial_conditions:
        value_func = ic_cond['value']
        var_component = problem.get_variable_index(ic_cond['variable'])
        
        dde_ic = dde.IC(geom, value_func, lambda _, on_initial: on_initial, component=var_component)
        dde_ic_list.append(dde_ic)
    return dde_ic_list


def create_dde_geometry(problem: ProblemDefinition):
    """
    Creates a generalized dde.geometry object from the problem definition.
    """
    coords = problem.domain_coords
    
    # Check if the domain is time-dependent
    is_time_dependent = 't' in coords
    
    # For a purely spatial problem (like steady-state), use Hypercube directly.
    # For a time-dependent problem, the geometry for BCs is spatial,
    # and the time domain is separate.
    
    # DeepXDE's TimePDE handles the time domain separately, so the 'geom'
    # argument should only represent the SPATIAL domain.
    spatial_coords = {key: val for key, val in coords.items() if key != 't'}
    
    if not spatial_coords:
        raise ValueError("Problem must have at least one spatial dimension.")

    # These lists are created dynamically from the keys
    xmin = [v[0] for v in spatial_coords.values()]
    xmax = [v[1] for v in spatial_coords.values()]
    
    # Use the general Hypercube for the spatial domain
    spatial_geom = dde.geometry.Hypercube(xmin, xmax)

    if is_time_dependent:
        time_domain = dde.geometry.TimeDomain(coords['t'][0], coords['t'][1])
        # The final geometry is a product of the spatial and temporal domains
        return dde.geometry.GeometryXTime(spatial_geom, time_domain)
    else:
        # If no time, just return the spatial geometry
        return spatial_geom


def create_dde_data(problem: ProblemDefinition, training_params: dict):
    """
    Creates a dde.data.PDE or dde.data.TimePDE object from a ProblemDefinition.

    Args:
        problem (ProblemDefinition): The abstract definition of the problem.
        training_params (dict): A dictionary with keys like 'num_domain',
            'num_boundary', 'num_initial', etc.

    Returns:<
        A dde.data.PDE or dde.data.TimePDE object ready for model training.
    """
    print(f"--- Creating DDE data for problem: '{problem.name}' ---")
    
    # Step 1: Create the appropriate geometry (Hypercube or GeometryXTime)
    geom = create_dde_geometry(problem)
    print(f"Generated geometry of type: {type(geom).__name__}")
    
    # Step 2: Generate the boundary conditions
    bcs = create_dde_bcs(problem, geom)
    print(f"Generated {len(bcs)} boundary conditions.")

    # Step 3: Decide which data object to create based on time-dependence
    if 't' in problem.domain_coords:
        # Time-dependent problem
        print("Problem is time-dependent. Creating dde.data.TimePDE.")
        
        # Step 3a: Generate initial conditions
        ics = create_dde_ics(problem, geom)
        print(f"Generated {len(ics)} initial conditions.")

        # Step 3b: Combine conditions and create TimePDE object
        # The 'bcs' and 'ics' keyword arguments are the clearest way to do this
        data = dde.data.TimePDE(
            geom,
            lambda x,y : problem.residual_func(x, y, problem.scale), 
            [*ics, *bcs],  
            num_domain=training_params.get('num_domain', 1000),
            num_boundary=training_params.get('num_boundary', 500),
            num_initial=training_params.get('num_initial', 200),
            #anchors=None, # Add anchors if needed from problem def
            #solution=training_params.get('solution', None)
        )
    else:
        # Steady-state problem
        print("Problem is steady-state. Creating dde.data.PDE.")
        
        # Create PDE object
        data = dde.data.PDE(
            geom,
            lambda x,y : problem.residual_func(x, y, problem.scale),
            bcs,
            num_domain=training_params.get('num_domain', 1000),
            num_boundary=training_params.get('num_boundary', 500),
            #solution=training_params.get('solution', None)
        )
        
    return data

from process.mechanic.residual import pde_2d_residual
from process.heat.residual import lp_residual
from process.heat.scale import Scale
from domain_vars import transient_heat_2d_domain

#beam_problem = ProblemDefinition(
#    name="2D Plain Strain Beam",
#    description="2D plane strain.",
#    domain_coords={'x': [0.0, 1.0], 'y': [0.0, 10.0]},
#    variables=['u', 'v'],
#    residual_func=pde_2d_residual,
#    #scale=Scale()
#    boundary_conditions=[
#        {'type': 'Dirichlet', 'variable': 'u', 'value': 0.0, 'location_on': {'x': 'min'}},
#        {'type': 'Dirichlet', 'variable': 'v', 'value': 0.0, 'location_on': {'x': 'min'}},
#    ]
#)
heat_problem = ProblemDefinition(
    name="2D Transient Heat",
    description="Heat equation in a 2D domain over time.",
    domain_coords={'x': [0.0, 0.5], 'y': [0.0, 1.0], 't': [0.0, 8.6e4]},
    variables=['T'],

    residual_func=lp_residual,
    scale= Scale(transient_heat_2d_domain),

    boundary_conditions=[
        {'type': 'Dirichlet', 'variable': 'T', 'value': 100.0, 'location_on': {'x': 'min'}},
        {'type': 'Dirichlet', 'variable': 'T', 'value': 0.0, 'location_on': {'x': 'max'}},
    ],
    
    initial_conditions=[
        #{'type': 'IC', 'variable': 'T', 'value': lambda x: np.sin(np.pi * x[:, 0:1])}
        {'type': 'IC', 'variable': 'T', 'value': lambda x: 0.0}
    ]
)

