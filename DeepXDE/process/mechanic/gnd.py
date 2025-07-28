import math
import numpy as np
import dolfinx as df
import ufl
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.fem.petsc import LinearProblem
from utils.COMSOL import extract_static_displacement_data_einspannung
from FEM.output import evaluate_solution_at_points_on_rank_0, initialize_point_evaluation, load_fem_results, save_fem_results
from FEM.init_helper import create_dirichlet_bcs, create_mesh_and_function_space, create_solver
from material import concreteData

materialData = concreteData


def analytical_solution_FLL(x, q=1, L=1, EI=1):
    return (1/12)*x**3 - (1/24) * x**4 - (1/24) * x
def analytical_solution_FES(x, q=1, L=1, EI=1):
    return -(1/2)*q*L*x**2 + (1/6)*q*L*x**3
def analytical_solution_FLL_t(x,t):
    return np.sin(x)*np.cos(4*math.pi*t)

base_mapping = {
    'fest_los': analytical_solution_FLL,
    'einspannung': analytical_solution_FES,
    'fest_los_t': analytical_solution_FLL_t,
    #'einspannung_2d': solution_2d_einspannung
    }

    
def strain(u):
    return 0.5 * (ufl.grad(u) + ufl.grad(u).T)
def stress(u_vec, C):
    eps_u_voigt = ufl.as_vector([strain(u_vec)[0, 0], strain(u_vec)[1, 1], 2 * strain(u_vec)[0, 1]])
    sigma_voigt = ufl.dot(C, eps_u_voigt)
    return ufl.as_tensor([[sigma_voigt[0], sigma_voigt[2]],
                          [sigma_voigt[2], sigma_voigt[1]]])


def get_einspannung_weak_form(V, C):
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f_body = df.fem.Constant(V.mesh, PETSc.ScalarType((0, -materialData.rho * materialData.g)))

    a = ufl.inner(stress(u, C), strain(v)) * ufl.dx
    L = ufl.dot(f_body, v) * ufl.dx
    return a, L
def clamped_boundary_condition(x, x_min):
    return np.isclose(x[0], x_min)
     
def get_einspannung_2d_fem(domain_vars
                           ):
    """
    Calculates the displacement of a 2D clamped beam using FEM helpers.
    """
    comm = MPI.COMM_WORLD
    x_min, x_max = domain_vars.spatial['x']
    
    y_min, y_max = domain_vars.spatial['y']
    nx, ny = domain_vars.resolution['x'], domain_vars.resolution['y']

    # MESH
    element_desc = {"family": "Lagrange", "degree": 1, "type": "vector"}
    mesh, V = create_mesh_and_function_space(
        comm,
        domain_extents=[[x_min, y_min], [x_max, y_max]],
        domain_resolution=[nx, ny],
        element_desc=element_desc
    )

    # DEBUG: Check material properties
    print(f"DEBUG: rho = {materialData.rho} kg/m³")
    print(f"DEBUG: g = {materialData.g} m/s²")
    print(f"DEBUG: E = {materialData.E} Pa")
    print(f"DEBUG: Body force = {materialData.rho * materialData.g} N/m³")
    
    # Calculate expected gravity deflection for comparison
    L = x_max - x_min
    expected_gravity_deflection = (materialData.rho * materialData.g * L**4) / (8 * materialData.E)
    print(f"DEBUG: Expected gravity deflection ~{expected_gravity_deflection:.8f} m")
    

    # BC
    #bc_definitions = [
    #    {"where": lambda x: np.isclose(x[0], x_min), "value": (0.0, 0.0)}
    #]
    bc_definitions = [
        {"where": lambda x: np.isclose(x[0], x_min), "value": (0.0, 0.0)},
        {"where": lambda x: np.isclose(x[0], x_max), "value": -0.01, "subspace_idx": 1} # mayybe 0
    ]


    bcs = create_dirichlet_bcs(V, bc_definitions)

    #  MABYE DIFFRENT VALS
    #f_body = df.fem.Constant(mesh, PETSc.ScalarType((0, -materialData.rho*materialData.g)))
    C_np = materialData.C_stiffness_matrix().cpu().numpy()
    #print(f"DEBUG: Using g = {materialData.g} m/s^2")
    #print(f"DEBUG: Using rho = {materialData.rho} kg/m^3")
    #print("DEBUG: Stiffness matrix C (in Pa):\n", C_np)
    C = df.fem.Constant(mesh, np.asarray(C_np, dtype=PETSc.ScalarType))

    a, L = get_einspannung_weak_form(V, C)

    # SOLVER
    solver_function = create_solver(mesh, a, L, bcs, problem_type="linear")
    u_sol = solver_function()
    u_sol.name = "Displacement"

    if comm.rank == 0:
        u_array = u_sol.x.array.reshape(-1, 2)  # [n_nodes, 2]
        max_disp_x = np.max(np.abs(u_array[:, 0]))
        max_disp_y = np.max(np.abs(u_array[:, 1]))
        min_disp_y = np.min(u_array[:, 1])
        
        print(f"DEBUG: Max |displacement_x|: {max_disp_x:.6f} m")
        print(f"DEBUG: Max |displacement_y|: {max_disp_y:.6f} m") 
        print(f"DEBUG: Min displacement_y: {min_disp_y:.6f} m")
        
        # Check if prescribed displacement is applied
        #if abs(min_disp_y + 0.01) < 1e-6:
        #    print("✓ Prescribed displacement BC correctly applied!")
        #else:
        #    print("✗ Prescribed displacement BC NOT applied correctly!")
    
    return u_sol
    



def get_sigma_fem(u_sol, domain_vars, grid_resolution=(10, 10)):
    """
    Compute stress fields using the comprehensive helper system
    """
    # Get the mesh from the displacement solution
    mesh = u_sol.function_space.mesh
    comm = mesh.comm
    
    # Define element description for scalar stress fields
    scalar_element_desc = {
        "family": "Lagrange", 
        "degree": 1, 
        "type": "scalar",
        "name": "stress"
    }
    domain_extents = [[domain_vars.spatial['x'][0], domain_vars.spatial['y'][0]],
                      [domain_vars.spatial['x'][1], domain_vars.spatial['y'][1]]]
    # Create scalar function space using the helper
    _, V_scalar = create_mesh_and_function_space(
        comm, 
        domain_extents, 
        grid_resolution,
        scalar_element_desc
    )
    
    # But we need to use the SAME mesh, so create function space directly
    V_scalar = df.fem.functionspace(mesh, ("Lagrange", 1))
    
    # Material properties using your material system
    C_np = materialData.C_stiffness_matrix().cpu().numpy()
    C = df.fem.Constant(mesh, np.asarray(C_np, dtype=PETSc.ScalarType))

    # Compute strain and stress expressions
    eps_u_voigt = ufl.as_vector([
        strain(u_sol)[0, 0],      # epsilon_xx
        strain(u_sol)[1, 1],      # epsilon_yy  
        2 * strain(u_sol)[0, 1]   # gamma_xy
    ])
    sigma_voigt = ufl.dot(C, eps_u_voigt)
    
    # Define stress components
    stress_components = [
        {"expr": sigma_voigt[0], "name": "sigma_xx"},
        {"expr": sigma_voigt[1], "name": "sigma_yy"}, 
        {"expr": sigma_voigt[2], "name": "tau_xy"}
    ]
    
    # Create solver for each stress component using the helper
    stress_functions = []
    for comp in stress_components:
        # Define weak form for L2 projection
        a_form = ufl.inner(ufl.TrialFunction(V_scalar), ufl.TestFunction(V_scalar)) * ufl.dx
        L_form = ufl.inner(comp["expr"], ufl.TestFunction(V_scalar)) * ufl.dx
        
        # Create solver using the helper function
        solver_func = create_solver(
            mesh, 
            a_form, 
            L_form, 
            bcs=[],  # No boundary conditions for stress projection
            problem_type="linear"
        )
        
        # Solve for this stress component
        stress_func = solver_func()
        stress_func.name = comp["name"]
        stress_functions.append(stress_func)
    
    return stress_functions[0], stress_functions[1], stress_functions[2]
def get_ensemble_einspannung_2d_fem_points(domain_vars, points_data, comm):
    """
    Enhanced ensemble solution using helper functions, evaluated at specific points.
    Returns an array of shape (n_points, 8) with [u, v, exx, eyy, exy, sigma_xx, sigma_yy, tau_xy].
    """
    
    try:
        # 1. Get displacement and stress solutions
        u_sol = get_einspannung_2d_fem(domain_vars)
        sigma_xx, sigma_yy, tau_xy = get_sigma_fem(u_sol, domain_vars)
        
        # 2. Prepare for point evaluation
        mesh = u_sol.function_space.mesh
        _perform_eval, eval_points_3d, bb_tree = initialize_point_evaluation(
            mesh, points_data['spatial_points_flat'], comm
        )

        # 3. Evaluate displacement components (u, v)
        V = u_sol.function_space
        V_x, dof_map_x = V.sub(0).collapse()
        V_y, dof_map_y = V.sub(1).collapse()
        u_x_func = df.fem.Function(V_x)
        u_y_func = df.fem.Function(V_y)
        u_x_func.x.array[:] = u_sol.x.array[dof_map_x]
        u_y_func.x.array[:] = u_sol.x.array[dof_map_y]
        
        u_x_flat = evaluate_solution_at_points_on_rank_0(u_x_func, eval_points_3d, bb_tree, mesh, comm)
        u_y_flat = evaluate_solution_at_points_on_rank_0(u_y_func, eval_points_3d, bb_tree, mesh, comm)

        # 3.5. COMPUTE AND EVALUATE STRAIN COMPONENTS (NEW FOR V2)
        # Create scalar function space for strain fields
        V_scalar = df.fem.functionspace(mesh, ("Lagrange", 1))
        
        # Define strain expressions
        eps_xx_expr = strain(u_sol)[0, 0]
        eps_yy_expr = strain(u_sol)[1, 1]
        eps_xy_expr = strain(u_sol)[0, 1]  # Note: not 2* for engineering strain
        
        # Project strain components to scalar functions
        strain_components = [eps_xx_expr, eps_yy_expr, eps_xy_expr]
        strain_functions = []
        
        for strain_expr in strain_components:
            # Define weak form for L2 projection
            a_form = ufl.inner(ufl.TrialFunction(V_scalar), ufl.TestFunction(V_scalar)) * ufl.dx
            L_form = ufl.inner(strain_expr, ufl.TestFunction(V_scalar)) * ufl.dx
            
            # Create solver
            solver_func = create_solver(mesh, a_form, L_form, bcs=[], problem_type="linear")
            strain_func = solver_func()
            strain_functions.append(strain_func)
        
        # Evaluate strain components at points
        eps_xx_flat = evaluate_solution_at_points_on_rank_0(strain_functions[0], eval_points_3d, bb_tree, mesh, comm)
        eps_yy_flat = evaluate_solution_at_points_on_rank_0(strain_functions[1], eval_points_3d, bb_tree, mesh, comm)
        eps_xy_flat = evaluate_solution_at_points_on_rank_0(strain_functions[2], eval_points_3d, bb_tree, mesh, comm)

        # 4. Evaluate stress components
        sigma_xx_flat = evaluate_solution_at_points_on_rank_0(sigma_xx, eval_points_3d, bb_tree, mesh, comm)
        sigma_yy_flat = evaluate_solution_at_points_on_rank_0(sigma_yy, eval_points_3d, bb_tree, mesh, comm)
        tau_xy_flat = evaluate_solution_at_points_on_rank_0(tau_xy, eval_points_3d, bb_tree, mesh, comm)

        # 5. Combine results on rank 0 - NOW WITH 8 COMPONENTS FOR V2
        if comm.rank == 0:
            ensemble_values_flat = np.hstack([
                u_x_flat[:, np.newaxis],      # Component 0: ux
                u_y_flat[:, np.newaxis],      # Component 1: uy
                eps_xx_flat[:, np.newaxis],   # Component 2: exx
                eps_yy_flat[:, np.newaxis],   # Component 3: eyy
                eps_xy_flat[:, np.newaxis],   # Component 4: exy
                sigma_xx_flat[:, np.newaxis], # Component 5: sigma_xx
                sigma_yy_flat[:, np.newaxis], # Component 6: sigma_yy
                tau_xy_flat[:, np.newaxis]    # Component 7: tau_xy
            ])
            
            # Check reshape function exists
            if 'reshape_static_to_grid' in points_data:
                ensemble_values_at_points = points_data['reshape_static_to_grid'](ensemble_values_flat)
            else:
                print("WARNING: reshape_static_to_grid not found, returning flat array")
                ensemble_values_at_points = ensemble_values_flat
                
            return ensemble_values_at_points
        else:
            return None
            
    except Exception as e:
        print(f"Error in get_ensemble_einspannung_2d_fem_points: {e}")
        if comm.rank == 0:
            return None
        else:
            return None


def get_einspannung_2d_fem_points(domain_vars, points_data, comm):
    #result = load_fem_results('BASELINE/mechanic/2d/einspannung.npy')
    #if result is not None:
    #    return result
    GROUND = get_einspannung_2d_fem(domain_vars)
    #GROUND = load_fem_results("BASELINE/mechanic/2d/ground_truth.npy")
    
    domain = GROUND.function_space.mesh
    _perform_eval, eval_points_3d, bb_tree = initialize_point_evaluation(
        domain, points_data['spatial_points_flat'], comm
    )
        ### FIXED EVAL
    V = GROUND.function_space
    V_x, dof_map_x = V.sub(0).collapse()
    V_y, dof_map_y = V.sub(1).collapse()

    u_x_func = df.fem.Function(V_x)
    u_y_func = df.fem.Function(V_y)

    u_x_func.x.array[:] = GROUND.x.array[dof_map_x]
    u_y_func.x.array[:] = GROUND.x.array[dof_map_y]

    gt_u_x_flat = evaluate_solution_at_points_on_rank_0(u_x_func, eval_points_3d, bb_tree, domain, comm)
    gt_u_y_flat = evaluate_solution_at_points_on_rank_0(u_y_func, eval_points_3d, bb_tree, domain, comm)

    if comm.rank == 0:
        ground_values_at_points_flat = np.hstack((gt_u_x_flat[:, np.newaxis], gt_u_y_flat[:, np.newaxis]))
        ground_values_at_points = points_data['reshape_static_to_grid'](ground_values_at_points_flat)
    else:
        ground_values_at_points = None
    
    #save_fem_results("BASELINE/mechanic/2d/einspannung_with_u.npy", ground_values_at_points)

    return ground_values_at_points


def get_near_zero_ground(domain_vars, point_data, comm):
    data_array, coords, x_coords, y_coords = extract_static_displacement_data_einspannung(
        "BASELINE/mechanic/2d/no_body.vtu"
    )
    print('data_array_shape', data_array.shape, data_array.min().item(), data_array.max().item())
    return data_array