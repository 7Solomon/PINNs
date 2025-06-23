import dolfinx as df
import numpy as np
import torch
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile

from FEM.output import evaluate_solution_at_points_on_rank_0, initialize_point_evaluation
from FEM.init_helper import create_dirichlet_bcs, create_mesh_and_function_space, create_solver, get_dt, initialize_fem_state
from FEM.run_helper import execute_transient_simulation
from material import concreteData


def ufl_S_e(h, material):
    """Effective saturation (van Genuchten model) for UFL."""
    # Richards equation is typically for h < 0 (unsaturated). For h >= 0, Se = 1.
    h_abs = ufl.algebra.Abs(h)
    core = (1 + (material.alpha_vg * h_abs)**material.n_vg)**(-material.m_vg)
    return ufl.conditional(h < 0, core, 1.0)

def ufl_specific_moisture_capacity(h, material):
    """Specific moisture capacity C(h) = d(theta)/dh for UFL."""
    # C(h) is non-zero only for h < 0.
    h_abs = ufl.algebra.Abs(h)
    term1 = (material.theta_s - material.theta_r)
    term2 = material.alpha_vg * material.n_vg * material.m_vg
    term3 = (material.alpha_vg * h_abs)**(material.n_vg - 1)
    term4 = (1 + (material.alpha_vg * h_abs)**material.n_vg)**(-(material.m_vg + 1))
    C_unsat = term1 * term2 * term3 * term4
    # Return a small positive value for h >= 0 to avoid numerical issues.
    return ufl.conditional(h < 0, C_unsat, 1e-9)

def ufl_hydraulic_conductivity(h, material):
    """Hydraulic conductivity K(h) for UFL."""
    Se = ufl_S_e(h, material)
    # Clamp Se slightly away from 1 to avoid issues in the next term
    Se_clamped = ufl.min_value(Se, 1.0 - 1e-9)
    K_r = Se_clamped**0.5 * (1 - (1 - Se_clamped**(1 / material.m_vg))**material.m_vg)**2
    return material.K_s * K_r

def define_moisture_1d_head_weak_form(V,dt, uh, un, material):
    v = ufl.TestFunction(V)
    C = ufl_specific_moisture_capacity(uh, material)
    K = ufl_hydraulic_conductivity(uh, material)

    F = C * (uh - un) * v * ufl.dx + dt * ufl.dot(K * ufl.grad(uh), ufl.grad(v)) * ufl.dx
    return F 

    ## With grav
    #e_z = ufl.as_vector([1.0])
    #total_grad_h = ufl.grad(uh) + e_z
    #F = C * (uh - un) * v * ufl.dx + dt * ufl.dot(K * total_grad_h, ufl.grad(v)) * ufl.dx
    #return F

def define_heat_equation_forms(V, dt, alpha, un):
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = u * v * ufl.dx + dt * alpha * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = un * v * ufl.dx
    return df.fem.form(a), df.fem.form(L)


def get_richards_1d_head_fem(domain_vars,
                            nz=25,
                            evaluation_times: np.ndarray = None, 
                            evaluation_spatial_points_z: np.ndarray = None
                            ):
    """
    Runs a 1D transient FEM simulation for the Richards equation.
    """
    material = concreteData 


    # 1. MESH
    comm = MPI.COMM_WORLD
    z_min, z_max = domain_vars.spatial['z']
    t_min, t_max = domain_vars.temporal['t']
    
    element_desc = {"type": "scalar", "family": "Lagrange", "degree": 1}
    mesh, V = create_mesh_and_function_space(comm=comm, 
                                            domain_extents=[z_min, z_max], 
                                            domain_resolution=nz, 
                                            element_desc=element_desc
                                        )
    
    # 2. Get dt
    dt_fem_internal = get_dt(comm, evaluation_times)


    ### BC And IC
    bc_left_value = -1.0
    bc_right_value = -7.0
    #ef initial_condition_func(x):
    #   return bc_left_value + (bc_right_value - bc_left_value) * (x[0] - z_min) / (z_max - z_min)
    
    initial_conditions = {'head': -3.5}
    constants_def = {"dt": dt_fem_internal}
    state_vars = ["uh_head", "un_head"]
    fem_states, fem_constants = initialize_fem_state(
        V,
        initial_conditions=initial_conditions,
        element_desc=element_desc,
        constants_def=constants_def,
        state_vars=state_vars
    )

    uh = fem_states["uh_head"]
    un = fem_states["un_head"]
    dt_const = fem_constants["dt"]
    
    # Set the initial condition for the previous step as well
    un.x.array[:] = uh.x.array

    F = define_moisture_1d_head_weak_form(V, dt_const, uh, un, material)

    bcs = [
            {"where": lambda x: np.isclose(x[0], z_min), "value": bc_left_value, "component": None},
            {"where": lambda x: np.isclose(x[0], z_max), "value": bc_right_value, "component": None}
        ]
    bcs = create_dirichlet_bcs(V, bcs)

    #A, b_vec, compiled_a, compiled_L = _compile_forms_and_assemble_matrix(domain, a_form, L_form_template, un, bcs)
    #solver = _create_ksp_solver(domain, A)
    solver_function = create_solver(mesh, F, None, bcs, 'nonlinear', uh=uh)

    uh, final_evaluated_data = execute_transient_simulation(
        domain=mesh,
        t_start=t_min,
        t_end=t_max,
        dt_initial=dt_fem_internal,
        solver_function=solver_function,
        problem_type="nonlinear",
        fem_states=fem_states,
        fem_constants=fem_constants,
        evaluation_times=evaluation_times,
        evaluation_spatial_points_xy=evaluation_spatial_points_z
    )
    return uh, final_evaluated_data

def ufl_get_volumetric_water_content(S_total, material):
    """Calculates volumetric water content (theta) from total saturation (S)."""
    return S_total * (material.theta_s - material.theta_r) + material.theta_r

def ufl_get_effective_saturation(S_total, material):
    """Calculates effective saturation (Se) from total saturation (S)."""
    theta = ufl_get_volumetric_water_content(S_total, material)
    return ufl.max_value(0.0, (theta - material.theta_r) / (material.theta_s - material.theta_r))

def ufl_get_pressure_head(Se, material):
    """Calculates pressure head (h) from effective saturation (Se)."""
    # Clamp Se to avoid math domain errors (e.g., log(0), x**y where x<0)
    Se_clamped = ufl.min_value(ufl.max_value(Se, 1e-9), 1.0 - 1e-9)
    # van Genuchten-Mualem inverse model
    term = (Se_clamped**(-1.0 / material.m_vg)) - 1.0
    # Clamp term to be non-negative for the root
    term_clamped = ufl.max_value(term, 0.0)
    h_abs = (term_clamped**(1.0 / material.n_vg)) / material.alpha_vg
    # h is negative in unsaturated zone
    return -h_abs

def ufl_get_hydraulic_conductivity_from_Se(Se, material):
    """Calculates hydraulic conductivity (K) from effective saturation (Se)."""
    Se_clamped = ufl.min_value(ufl.max_value(Se, 1e-9), 1.0 - 1e-9)
    K_r = Se_clamped**0.5 * (1.0 - (1.0 - Se_clamped**(1.0 / material.m_vg))**material.m_vg)**2
    return material.K_s * K_r

def define_moisture_1d_saturation_weak_form(V, dt, uh, un, material):

    v = ufl.TestFunction(V)

    theta_h = ufl_get_volumetric_water_content(uh, material)
    Se_h = ufl_get_effective_saturation(uh, material)
    h_h = ufl_get_pressure_head(Se_h, material)
    K_h = ufl_get_hydraulic_conductivity_from_Se(Se_h, material)

    theta_n = ufl_get_volumetric_water_content(un, material)

    # (theta_h - theta_n)/dt * v + K(h) * grad(h) * grad(v) = 0
    F = (theta_h - theta_n) * v * ufl.dx + dt * ufl.dot(K_h * ufl.grad(h_h), ufl.grad(v)) * ufl.dx
    return F


def get_richards_1d_saturation_fem(domain_vars,
                                   nz=25,
                                   evaluation_times: np.ndarray = None,
                                   evaluation_spatial_points_z: np.ndarray = None
                                   ):
    """
    Runs a 1D transient FEM simulation for the Richards equation in saturation form.
    """
    material = concreteData

    # 1. MESH
    comm = MPI.COMM_WORLD
    z_min, z_max = domain_vars.spatial['z']
    t_min, t_max = domain_vars.temporal['t']

    element_desc = {"type": "scalar", "family": "Lagrange", "degree": 1, "name": "saturation"}
    mesh, V = create_mesh_and_function_space(comm, [z_min, z_max], nz, element_desc=element_desc)

    # Dt
    dt_fem_internal = get_dt(comm, evaluation_times)

    # BC and IC (values for total saturation S)
    bc_left_value = 0.9
    bc_right_value = 0.2
    initial_condition_value = 0.5

    initial_conditions = {'saturation': initial_condition_value}
    constants_def = {"dt": dt_fem_internal}
    state_vars = ["uh_saturation", "un_saturation"]
    fem_states, fem_constants = initialize_fem_state(
        V,
        initial_conditions=initial_conditions,
        element_desc=element_desc,
        state_vars=state_vars,
        constants_def=constants_def,
    )

    uh = fem_states["uh_saturation"]
    un = fem_states["un_saturation"]
    dt_const = fem_constants["dt"]

    # Set the initial condition for the previous step as well
    un.x.array[:] = uh.x.array

    # weak Form
    F = define_moisture_1d_saturation_weak_form(V, dt_const, uh, un, material)

    # BCs
    bcs_defs = [
        {"where": lambda x: np.isclose(x[0], z_min), "value": bc_left_value, "component": None},
        {"where": lambda x: np.isclose(x[0], z_max), "value": bc_right_value, "component": None}
    ]
    bcs = create_dirichlet_bcs(V, bcs_defs)

    # SOL
    solver_function = create_solver(mesh, F, None, bcs, 'nonlinear', uh=uh)

    # LOOP to get f
    uh, final_evaluated_data = execute_transient_simulation(
        domain=mesh,
        t_start=t_min,
        t_end=t_max,
        dt_initial=dt_fem_internal,
        solver_function=solver_function,
        problem_type="nonlinear",
        fem_states=fem_states,
        fem_constants=fem_constants,
        evaluation_times=evaluation_times,
        evaluation_spatial_points_xy=evaluation_spatial_points_z
    )
    return uh, final_evaluated_data
