import basix
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import numpy as np
import os

import dolfinx
from dolfinx import fem, mesh, io, nls, log, geometry # Added geometry
from dolfinx.fem.petsc import NonlinearProblem
#from dolfinx.nls.petsc import NewtonSolver

from FEM.init_helper import create_dirichlet_bcs, create_mesh_and_function_space, create_solver, get_dt, initialize_fem_state
from FEM.output import evaluate_solution_at_points_on_rank_0, initialize_point_evaluation
from FEM.run_helper import execute_transient_simulation
from material import concreteData # Ensure this contains all necessary parameters

materialData = concreteData

def epsilon_strain(u_func):
    """Computes the strain tensor: epsilon = 0.5 * (grad(u) + grad(u).T)"""
    return ufl.sym(ufl.grad(u_func))

def epsilon_vol(u_func):
    """Computes volumetric strain: tr(epsilon) or div(u) for small strains."""
    return ufl.div(u_func)

def effective_stress_sigma(u_func, E_const, nu_const, dim):
    """Computes the effective stress tensor sigma_eff = lambda*tr(eps)*I + 2*mu*eps."""
    lmbda = E_const * nu_const / ((1 + nu_const) * (1 - 2 * nu_const))
    mu = E_const / (2 * (1 + nu_const))
    eps = epsilon_strain(u_func)
    return lmbda * ufl.tr(eps) * ufl.Identity(dim) + 2 * mu * eps

def soil_water_retention_curve_head(theta_func, mat_data):
    """Computes capillary head (negative for suction) using Van Genuchten model."""
    theta_r = fem.Constant(theta_func.ufl_domain(), ScalarType(getattr(mat_data, 'theta_r', 0.05)))
    theta_s = fem.Constant(theta_func.ufl_domain(), ScalarType(getattr(mat_data, 'theta_s', 0.40)))
    alpha_vg = fem.Constant(theta_func.ufl_domain(), ScalarType(getattr(mat_data, 'alpha_vg', 0.1)))
    m_vg = fem.Constant(theta_func.ufl_domain(), ScalarType(getattr(mat_data, 'm_vg', 0.5)))
    n_vg = fem.Constant(theta_func.ufl_domain(), ScalarType(getattr(mat_data, 'n_vg', 2.0)))
    
    # Clamp s_eff to avoid issues at limits, e.g., log(0) or powers of non-positive numbers
    s_eff_eps = 1e-6
    s_eff_raw = (theta_func - theta_r) / (theta_s - theta_r)
    
    # Clamp s_eff between [eps, 1-eps] using ufl.conditional
    # s_eff = ufl.Max(ufl.Min(s_eff_raw, 1.0 - s_eff_eps), s_eff_eps)
    s_eff_min_clipped = ufl.conditional(ufl.lt(s_eff_raw, ScalarType(1.0 - s_eff_eps)), s_eff_raw, ScalarType(1.0 - s_eff_eps))
    s_eff = ufl.conditional(ufl.gt(s_eff_min_clipped, ScalarType(s_eff_eps)), s_eff_min_clipped, ScalarType(s_eff_eps))
    
    term_in_pow = s_eff**(-1.0/m_vg) - 1.0
    # Ensure term_in_pow is non-negative for the outer power
    # term_in_pow_safe = ufl.Max(term_in_pow, ScalarType(0.0))
    head = (1.0 / alpha_vg) * (term_in_pow)**(1.0/n_vg)
    return head


def pore_pressure(theta_func, mat_data):
    """Computes pore pressure from moisture content via the SWRC."""
    # Note: The head is negative for unsaturated conditions (suction).
    # p_w = -rho_w * g * head. The PINN returns -head, so we match that.
    head = soil_water_retention_curve_head(theta_func, mat_data)
    rho_w = fem.Constant(theta_func.ufl_domain(), ScalarType(getattr(mat_data, 'rho_w', 1000.0)))
    g = fem.Constant(theta_func.ufl_domain(), ScalarType(getattr(mat_data, 'g', 9.81)))
    return rho_w * g * (-head)


def moisture_diffusivity_D(u_func, mat_data):
    """Computes moisture diffusivity as a function of volumetric strain."""
    D0 = fem.Constant(u_func.ufl_domain(), ScalarType(getattr(mat_data, 'D_moisture', 1e-10)))
    coupling_coef = fem.Constant(u_func.ufl_domain(), ScalarType(getattr(mat_data, 'strain_moisture_coulling_coef', 0.0)))
    return D0 * (1 + coupling_coef * epsilon_vol(u_func))


def get_variational_form(V, w, w_n, constants):
        """Creates the total variational form F(w; w_n) = 0 for the fully coupled problem."""
        w_test=ufl.TestFunction(V)
        u, theta = ufl.split(w)
        u_n, theta_n = ufl.split(w_n)
        v, q = ufl.split(w_test)

        dt = constants["dt"]
        E = constants["E"]
        nu = constants["nu"]
        dim = V.mesh.geometry.dim
        
        # --- Mechanical part ---
        p_w = pore_pressure(theta, materialData)
        alpha_biot = fem.Constant(V.mesh, ScalarType(materialData.alpha_biot))
        sigma_eff = effective_stress_sigma(u, E, nu, dim)
        sigma_total = sigma_eff - alpha_biot * p_w * ufl.Identity(dim)
        
        # Body force (gravity) as in the PINN residual
        rho_bulk = fem.Constant(V.mesh, ScalarType(1.3e4)) # Matching PINN's res_y
        f_body = ufl.as_vector([0, -rho_bulk])
        
        F_mech = ufl.inner(sigma_total, epsilon_strain(v)) * ufl.dx - ufl.inner(f_body, v) * ufl.dx

        # --- Moisture part ---
        # Time derivative of volumetric strain for coupling term
        eps_vol = epsilon_vol(u)
        eps_vol_n = epsilon_vol(u_n)
        deps_vol_dt = (eps_vol - eps_vol_n) / dt
        
        # Strain-dependent diffusivity
        D = moisture_diffusivity_D(u, materialData)
        
        # Moisture flux q_m = -D * grad(theta)
        moisture_flux = -D * ufl.grad(theta)
        
        # Weak form of: d(theta)/dt + alpha*d(eps_vol)/dt - div(D*grad(theta)) = 0
        F_moisture = (((theta - theta_n) / dt) * q + alpha_biot * deps_vol_dt * q - ufl.inner(moisture_flux, ufl.grad(q))) * ufl.dx
        
        F_total = F_mech + F_moisture
        J = ufl.derivative(F_total, w)
        return F_total, J




def get_coupled_transient_fem(domain_vars, 
                            num_elements_x=32, num_elements_y=16, dt_value=0.1,
                            evaluation_times: np.ndarray = None,
                            evaluation_spatial_points_xy: np.ndarray = None):
    """
    Solves the coupled transient poromechanics problem using DOLFINx.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    x_min, x_max = domain_vars.spatial['x']
    y_min, y_max = domain_vars.spatial['y']
    t_min, t_max = domain_vars.temporal['t']
    
    # --- 1. Use the helper for Mesh and Mixed Function Space creation ---
    element_description = {
        "type": "mixed",
        "elements": [
            {"family": "Lagrange", "degree": 2, "type": "vector", "name": "u"},    # DEGREE 2 wichtig für COUPLING da  "LADYHENADA BUBUZSKA"
            {"family": "Lagrange", "degree": 1, "type": "scalar", "name": "theta"}
        ]
    }
    mesh, V = create_mesh_and_function_space(
        comm, 
        domain_extents=[[x_min, y_min], [x_max, y_max]], 
        domain_resolution=[num_elements_x, num_elements_y], 
        element_desc=element_description
    )
    dim = mesh.geometry.dim
    dt_fem_internal = get_dt(comm, evaluation_times)

    theta_bottom_val = materialData.theta_r
    theta_top_val = 0.9 * materialData.theta_s
    def initial_theta_smooth(x):
        y_normalized = (x[1] - y_min) / (y_max - y_min)
        return theta_bottom_val + (theta_top_val - theta_bottom_val) * y_normalized

    ## Boundary conditions and initial conditions
    initial_conditions = {
        "u": lambda x: np.zeros((dim, x.shape[1])),
        "theta": initial_theta_smooth
    }
    constants_def = {"dt": dt_fem_internal, "E": materialData.E, "nu": materialData.nu}
    state_vars = ["uh_w", "un_w"]
    fem_states, fem_constants = initialize_fem_state(
        V,
        initial_conditions=initial_conditions,
        element_desc=element_description,
        constants_def=constants_def,
        state_vars=state_vars
    )

    uh = fem_states["uh_w"]
    un = fem_states["un_w"]
    un.x.array[:] = uh.x.array

    bcs_definition = [{
        "where": lambda x: np.isclose(x[0], x_min),
        "value": np.zeros(dim, dtype=ScalarType),
        "subspace_idx": 0  # Applied to displacement u
    },
    {
        "where": lambda x: np.isclose(x[1], y_max),
        "value": 0.9 * materialData.theta_s,
        "subspace_idx": 1  # Applied to moisture theta
    }]
    bcs = create_dirichlet_bcs(V, bcs_definition)

    F, J = get_variational_form(
        V,
        w=uh,
        w_n=un,
        constants={
            "dt": fem_constants["dt"],
            "E": fem_constants["E"],
            "nu": fem_constants["nu"],
        }
    )
    solver_function = create_solver(mesh, F, J, bcs, 'nonlinear', uh=uh, J_form=J)

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
        evaluation_spatial_points_xy=evaluation_spatial_points_xy
    )
    return uh, final_evaluated_data


    # --- 2. Define solution, trial, and test functions ---
    #w_sol = fem.Function(W)      # Current solution (u, theta)
    #w_n = fem.Function(W)        # Previous solution (u_n, theta_n)
    #w_trial = ufl.TrialFunction(W)
    #w_test = ufl.TestFunction(W)
    #
    ## Split functions into components
    #u, theta = ufl.split(w_sol)
    #u_n, theta_n = ufl.split(w_n)
    #v, q = ufl.split(w_test)
#
    ## --- 3. Define constants and initial conditions ---
    #dt = fem.Constant(domain, ScalarType(dt_value))
    #E = fem.Constant(domain, ScalarType(materialData.E))
    #nu = fem.Constant(domain, ScalarType(materialData.nu))
    #
    ## Initial conditions (e.g., zero displacement, initial moisture)
    #w_n.sub(0).interpolate(lambda x: np.zeros((dim, x.shape[1])))
    #w_n.sub(1).interpolate(lambda x: np.full(x.shape[1], materialData.theta_s * 0.9))
    #w_sol.x.array[:] = w_n.x.array
#
    ## --- 4. Define boundary conditions ---
    ## Mechanical BC: Clamp left boundary
    #def clamped_boundary(x):
    #    return np.isclose(x[0], x_min)
    #
    #W0, _ = W.sub(0).collapse()
    #u_bc_dofs = fem.locate_dofs_geometrical((W.sub(0), W0), clamped_boundary)
    #bc_u = fem.dirichletbc(np.zeros(dim, dtype=ScalarType), u_bc_dofs, W.sub(0))
#
    ## Moisture BC: Constant moisture at top boundary
    #def top_boundary(x):
    #    return np.isclose(x[1], y_max)
    #    
    #W1, _ = W.sub(1).collapse()
    #theta_bc_dofs = fem.locate_dofs_geometrical((W.sub(1), W1), top_boundary)
    #bc_theta = fem.dirichletbc(ScalarType(materialData.theta_s), theta_bc_dofs, W.sub(1))
    #
    #bcs = [bc_u, bc_theta]
#
    ## --- 5. Define the variational form ---
    ## Governing equations (implicit Euler)
    #p_w = pore_pressure(theta, materialData)
    #sigma_eff = effective_stress_sigma(u, E, nu, dim)
    #sigma_total = sigma_eff - p_w * ufl.Identity(dim)
    #
    ## Balance of momentum
    #F_mech = ufl.inner(sigma_total, epsilon_strain(v)) * ufl.dx
    #
    ## Balance of mass (moisture)
    #D = moisture_diffusivity_D(theta, materialData)
    #moisture_flux = -D * ufl.grad(p_w) # Darcy's law
    #F_moisture = ((theta - theta_n) / dt) * q * ufl.dx - ufl.inner(moisture_flux, ufl.grad(q)) * ufl.dx
    #
    #F_total = F_mech + F_moisture
#
    ## --- 6. Setup solver and time loop ---
    #problem = NonlinearProblem(F_total, w_sol, bcs=bcs)
    #solver = nls.petsc.NewtonSolver(comm, problem)
    #solver.convergence_criterion = "incremental"
    #
    ## Setup point evaluation
    #perform_eval, eval_points_3d, bb_tree = initialize_point_evaluation(
    #    domain, evaluation_spatial_points_xy, comm
    #)
    #all_evaluated_data_rank0 = []
#
    ## Time-stepping loop
    #t_current = t_min
    #for t_eval in evaluation_times:
    #    if rank == 0:
    #        print(f"Solving up to t = {t_eval:.2f}")
    #    
    #    while t_current < t_eval - 1e-9:
    #        t_current += dt_value
    #        solver.solve(w_sol)
    #        w_n.x.array[:] = w_sol.x.array
    #    
    #    # Evaluate solution at points
    #    if perform_eval:
    #        # Note: This evaluates both u and theta. You may need to adapt the post-processing.
    #        # For now, we evaluate the displacement part (sub(0)).
    #        u_sol, _ = w_sol.sub(0).collapse()
    #        eval_data = evaluate_solution_at_points_on_rank_0(u_sol, eval_points_3d, bb_tree, domain, comm)
    #        if rank == 0 and eval_data is not None:
    #            all_evaluated_data_rank0.append(eval_data)
#
    ## --- 7. Finalize and return ---
    #final_evaluated_data = None
    #if perform_eval and rank == 0:
    #    final_evaluated_data = np.array(all_evaluated_data_rank0) if all_evaluated_data_rank0 else np.array([])
    #    
    #return w_sol, final_evaluated_data