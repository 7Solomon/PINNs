
import numpy as np
from mpi4py import MPI

from FEM.init_helper import create_dirichlet_bcs, create_mesh_and_function_space, create_solver, get_dt, initialize_fem_state

import ufl
from dolfinx import fem

from FEM.run_helper import execute_transient_simulation
from material import concreteData


def strain(u_vec):
    return 0.5 * (ufl.grad(u_vec) + ufl.grad(u_vec).T)

def stress(u_vec, temp, thermal_expansion_coefficient, C):
    # Thermal strain
    eps_thermal = ufl.as_vector([thermal_expansion_coefficient * temp, thermal_expansion_coefficient * temp, 0])

    # Mechanical strain
    eps_u_voigt = ufl.as_vector([strain(u_vec)[0, 0], strain(u_vec)[1, 1], 2 * strain(u_vec)[0, 1]])
    
    # Elastic strain
    eps_elastic = eps_u_voigt - eps_thermal
    
    # Stress in Voigt notation
    sigma_voigt = ufl.dot(C, eps_elastic)
    
    return ufl.as_tensor([[sigma_voigt[0], sigma_voigt[2]],
                          [sigma_voigt[2], sigma_voigt[1]]])



def define_thermal_mechanical_weak_form(V, dt, uh, un, C, thermal_expansion_coefficient_const, alpha_thermal_diffusivity):
    (u, T) = ufl.TrialFunctions(V)
    (v, q) = ufl.TestFunctions(V)
    (u_n, T_n) = ufl.split(un)

    # mechanic weak
    sigma_u = stress(u, T, thermal_expansion_coefficient_const, C)
    a_mech = ufl.inner(sigma_u, strain(v)) * ufl.dx
    a_thermal = (T/dt * q + alpha_thermal_diffusivity * ufl.dot(ufl.grad(T), ufl.grad(q))) * ufl.dx
    
    a = a_mech + a_thermal
    
    L = (T_n / dt * q) * ufl.dx
    return a, L



def get_thermal_mechanical_fem(
        domain_vars,
        grid_resolution=(25,25),
        evaluation_times: np.ndarray = None, 
        evaluation_spatial_points_xy: np.ndarray = None
    ):
    
    comm = MPI.COMM_WORLD
    x_min, x_max = domain_vars.spatial['x']
    y_min, y_max = domain_vars.spatial['y']
    t_min, t_max = domain_vars.temporal['t']

    element_desc = {
        "type": "mixed",
        "elements": [
            {"type": "vector", "family": "Lagrange", "degree": 2, "name": "u"},
            {"type": "scalar", "family": "Lagrange", "degree": 1, "name": "temp"}
        ]
    }
    mesh, V = create_mesh_and_function_space(comm=comm, 
                                            domain_extents=[[x_min, y_min], [x_max, y_max]],
                                            domain_resolution=grid_resolution,
                                            element_desc=element_desc)
    
    dt_fem_internal = get_dt(comm, evaluation_times)

    # BCs
    temprature_left_value = 10.0
    temprature_right_value = 0.0
    initial_temperature_value = 0.5

    bcs = [
        {"where": lambda x: np.isclose(x[0], x_min), "value": temprature_left_value, "subspace_idx": 1},
        {"where": lambda x: np.isclose(x[0], x_max), "value": temprature_right_value, "subspace_idx": 1},
        {"where": lambda x: np.isclose(x[1], y_min), "value": (0.0, 0.0), "subspace_idx": 0},
    ]
    bcs = create_dirichlet_bcs(V, bcs)

    # Initial state
    initial_conditions = {
        'u' : (0.0, 0.0),
        'temp': initial_temperature_value
        }
    constants_def = {"dt": dt_fem_internal, 
                    'C': concreteData.C_stiffness_matrix().cpu().numpy(), 
                    'thermal_expansion_coefficient': concreteData.thermal_expansion_coefficient, 
                    'alpha_thermal_diffusivity': concreteData.alpha_thermal_diffusivity}
    state_vars = ["uh", "un"]
    fem_states, fem_constants = initialize_fem_state(
        V,
        initial_conditions=initial_conditions,
        element_desc=element_desc,
        constants_def=constants_def,
        state_vars=state_vars
    )

    # Extract
    uh = fem_states["uh"]
    un = fem_states["un"]
    dt_const = fem_constants["dt"]
    C_const = fem_constants["C"]
    thermal_expansion_coefficient_const = fem_constants["thermal_expansion_coefficient"]
    alpha_thermal_diffusivity_const = fem_constants["alpha_thermal_diffusivity"]

    # Initial and previos same in first 
    un.x.array[:] = uh.x.array

    # weak
    a, L = define_thermal_mechanical_weak_form(V, dt_const, uh, un, C_const, thermal_expansion_coefficient_const, alpha_thermal_diffusivity_const)

    solver_function = create_solver(mesh, a, L, bcs, 'linear', uh=uh)

    uh, final_evaluated_data = execute_transient_simulation(
        domain=mesh,
        t_start=t_min,
        t_end=t_max,
        dt_initial=dt_fem_internal,
        solver_function=solver_function,
        problem_type="linear",
        fem_states=fem_states,
        fem_constants=fem_constants,
        evaluation_times=evaluation_times,
        evaluation_spatial_points_xy=evaluation_spatial_points_xy
    )
    return uh, final_evaluated_data

    



