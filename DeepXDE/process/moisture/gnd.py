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
    mesh, V = create_mesh_and_function_space(comm, [z_min, z_max], nz, element_desc=element_desc)
    
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



    ############
    #############
    

    #dt = df.fem.Constant(mesh, PETSc.ScalarType(dt_fem_internal))
#
    ## 3. Define functions for solution
    #uh = df.fem.Function(V)
    #uh.name = "pressure_head"
    #un = df.fem.Function(V)
    #un.name = "h_previous"
#
    ## Set initial condition based on your PINN setup
    #initial_h = -3.5  # [m]
    #un.interpolate(lambda x: np.full(x.shape[1], initial_h))
    #uh.x.array[:] = un.x.array
#
    ## 4. Define variational problem (weak form)
    #v = ufl.TestFunction(V)
    #
    #C = ufl_specific_moisture_capacity(uh, material)
    #K = ufl_hydraulic_conductivity(uh, material)
#
    ## integral(C(uh)*(uh-un)/dt*v) + integral(K(uh)*grad(uh) . grad(v)) = 0
    ## no gravity term (dK/dz).
    #residual = C * (uh - un) * v * ufl.dx + dt * ufl.dot(K * ufl.grad(uh), ufl.grad(v)) * ufl.dx
    #
    ## 5. Boundary Conditions
    #left_h_val = -1.0  # [m]
    #right_h_val = -7.0 # [m]
#
    #left_dofs = df.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], z_min))
    #right_dofs = df.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], z_max))
#
    #bc_left = df.fem.dirichletbc(PETSc.ScalarType(left_h_val), left_dofs, V)
    #bc_right = df.fem.dirichletbc(PETSc.ScalarType(right_h_val), right_dofs, V)
    #bcs = [bc_left, bc_right]
#
    ## 6. Setup Non-linear Solver
    #problem = NonlinearProblem(residual, uh, bcs=bcs)
    #solver = NewtonSolver(comm, problem)
    #solver.convergence_criterion = "incremental"
    #solver.rtol = 1e-6
#
    ## 7. Setup output file for visualization in ParaView
    #xdmf = XDMFFile(comm, "richards_simulation.xdmf", "w")
    #xdmf.write_mesh(mesh)
    #xdmf.write_function(uh, t_min)
#
    ## 8. Time-stepping loop
    #t = t_min
    #if comm.rank == 0:
    #    print(f"Starting simulation from t={t_min} to t={t_max} with dt={dt_fem_internal}")
#
    #while t < t_max:
    #    t += dt_fem_internal
    #    
    #    try:
    #        num_its, converged = solver.solve(uh)
    #        if not converged:
    #            if comm.rank == 0:
    #                print(f"Newton solver did not converge at t={t}. Stopping.")
    #            break
    #        uh.x.scatter_forward()
    #    except RuntimeError as e:
    #        if comm.rank == 0:
    #            print(f"Solver failed at t={t} with error: {e}. Stopping.")
    #        break
    #        
    #    un.x.array[:] = uh.x.array
    #    xdmf.write_function(uh, t)
#
    #    if comm.rank == 0:
    #        print(f"t = {t:6.2f} | Newton iterations: {num_its}")
#
    #xdmf.close()
    #if comm.rank == 0:
    #    print("Simulation finished. Output saved to richards_simulation.xdmf")
    #return uh


