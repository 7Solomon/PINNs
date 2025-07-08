import dolfinx as df
import numpy as np
from petsc4py import PETSc
from dolfinx.fem import petsc
import ufl
from dolfinx import nls
import basix

#def _create_mesh_and_function_space(comm, x_min, x_max, y_min, y_max, Nx, Ny):
#    extent = [[x_min, y_min], [x_max, y_max]]
#    domain = df.mesh.create_rectangle(comm, extent, [Nx, Ny], df.mesh.CellType.quadrilateral)
#    V = df.fem.functionspace(domain, ("Lagrange", 1))
#    return domain, V

def create_mesh_and_function_space(comm, domain_extents, domain_resolution, element_desc):
    """
    Creates a mesh and a function space based on a description.

    Args:
        comm: MPI communicator.
        domain_extents: [[x_min, y_min], [x_max, y_max]].
        domain_resolution: [Nx, Ny].
        element_desc (dict): Describes the function space. 
                             E.g., {"family": "Lagrange", "degree": 1, "type": "scalar"}
                             E.g., {"family": "Lagrange", "degree": 2, "type": "vector"}
                             E.g., {"type": "mixed", "elements": [desc1, desc2]}
    """
    is_2d = isinstance(domain_extents[0], (list, tuple, np.ndarray))

    if is_2d:
        if not (isinstance(domain_resolution, (list, tuple, np.ndarray)) and len(domain_resolution) == 2):
            raise ValueError("For a 2D domain, domain_resolution must be a list/tuple of two integers (nx, ny).")
        mesh = df.mesh.create_rectangle(comm, domain_extents, domain_resolution, df.mesh.CellType.quadrilateral)
    else:
        if not isinstance(domain_resolution, int):
            raise ValueError("For a 1D domain, domain_resolution must be an integer (nx).")
        mesh = df.mesh.create_interval(comm, domain_resolution, domain_extents)
    
    dim = mesh.geometry.dim
    if element_desc["type"] == "scalar":
        V = df.fem.functionspace(mesh, (element_desc["family"], element_desc["degree"]))
    elif element_desc["type"] == "vector":
        V = df.fem.functionspace(mesh, (element_desc["family"], element_desc["degree"], (dim,)))
    elif element_desc["type"] == "mixed":
        ufl_elements = []
        for e_desc in element_desc["elements"]:
            elem_family = e_desc["family"]
            elem_degree = e_desc["degree"]
            cell_name = mesh.ufl_cell().cellname()
            if e_desc["type"] == "vector":
                ufl_elements.append(basix.ufl.element(elem_family, cell_name, elem_degree, shape=(dim,)))
            elif e_desc["type"] == "scalar":
                ufl_elements.append(basix.ufl.element(elem_family, cell_name, elem_degree))
            else:
                raise ValueError(f"Unsupported element type in mixed space: {e_desc['type']}")
        
        mixed_element = basix.ufl.mixed_element(ufl_elements)
        V = df.fem.functionspace(mesh, mixed_element)
    else:
        raise ValueError(f"Unknown element description type: {element_desc['type']}")
    return mesh, V


    ## Create the mixed UFL element
    #W_mixed_ufl_element = basix.ufl.MixedElement([Ue_ufl, Te_ufl])
    #
    ## Create the final mixed function space using the mixed UFL element
    ## and dolfinx.fem.functionspace (lowercase 'f')
    #W = fem.functionspace(domain, W_mixed_ufl_element)
    #
    ## Create separate function spaces and functions for evaluation
    #V_u, map_u = W.sub(0).collapse()
    #V_theta, map_theta = W.sub(1).collapse()
    #u_eval_func = fem.Function(V_u)
    #u_eval_func.name = "u_eval"
    #theta_eval_func = fem.Function(V_theta)
    #theta_eval_func.name = "theta_eval"


def initialize_fem_state(V, initial_conditions, element_desc, state_vars=["uh", "un"], constants_def=None):
    """
    Initializes state functions and constants. Handles mixed element spaces correctly.

    Args:
        V: The function space.
        initial_conditions (dict): Maps component name to its initial value.
                                   E.g., {"temperature": 20.0, "pressure_head": -5.0}
        element_desc (dict): The same description used to create V. Crucial for mixed spaces.
        state_vars (list): Names of the state functions to create (e.g., current, previous).
        constants_def (dict): Definitions for dolfinx.fem.Constant objects.
    """
    domain = V.mesh
    
    fem_states = {}
    for name in state_vars:  # e.g., name = "uh"
        func = df.fem.Function(V, name=name)
        
        is_mixed = V.element.num_sub_elements > 1
        
        if is_mixed and len(initial_conditions) == 1:
            # Get the single function provided, regardless of its key name.
            init_func_for_whole_space = list(initial_conditions.values())[0]
            if callable(init_func_for_whole_space):
                try:
                    func.interpolate(init_func_for_whole_space)
                    fem_states[name] = func
                    continue  # Skip to the next state_var (e.g., "un")
                except (RuntimeError, ValueError) as e:
                    if V.mesh.comm.rank == 0:
                        print(f"Warning: Failed to interpolate '{name}' with single function ({e}). Falling back to subspace initialization.")

        if is_mixed:
            # --- Mixed Space Initialization ---
            if element_desc.get("type") != "mixed":
                raise ValueError("Function space is mixed, but element_desc is not.")
            
            for i, sub_elem_desc in enumerate(element_desc["elements"]):
                sub_func = func.sub(i)
                key_name = sub_elem_desc.get("name")
                
                if key_name and key_name in initial_conditions:
                    val = initial_conditions[key_name]
                    
                    if callable(val):
                        sub_func.interpolate(val)
                    else:
                        V_sub, _ = V.sub(i).collapse()
                        shape = V_sub.value_shape
                        if not shape:  # Scalar subspace
                            sub_func.interpolate(lambda x: np.full(x.shape[1], val))
                        else:  # Vector subspace
                            const_vals = np.array(val, dtype=PETSc.ScalarType)
                            if const_vals.shape != shape:
                                raise ValueError(f"Initial value shape {const_vals.shape} for '{key_name}' does not match subspace shape {shape}.")
                            sub_func.interpolate(lambda x: np.outer(const_vals, np.ones(x.shape[1])))
                else:
                    if V.mesh.comm.rank == 0:
                        print(f"Warning: No initial condition for component '{key_name}' (subspace {i}) of '{name}'. Defaulting to zero.")
        else:
            # --- Standard (Scalar/Vector) Space Initialization ---
            key_name = element_desc.get("name")
            if not (key_name and key_name in initial_conditions) and len(initial_conditions) == 1:
                key_name = list(initial_conditions.keys())[0]

            if key_name and key_name in initial_conditions:
                val = initial_conditions[key_name]
                if callable(val):
                    func.interpolate(val)
                else:
                    shape = func.function_space.value_shape
                    if not shape:  # Scalar space
                        func.interpolate(lambda x: np.full(x.shape[1], val))
                    else:  # Vector space
                        const_vals = np.array(val, dtype=PETSc.ScalarType)
                        if const_vals.shape != shape:
                            raise ValueError(f"Initial value shape {const_vals.shape} for '{key_name}' does not match space shape {shape}.")
                        func.interpolate(lambda x: np.outer(const_vals, np.ones(x.shape[1])))
            else:
                if V.mesh.comm.rank == 0:
                    print(f"Warning: No initial condition found for state variable '{name}'. Defaulting to zero.")

        fem_states[name] = func
    
    # Create constants
    fem_constants = {}
    if constants_def:
        for name, value in constants_def.items():
            if isinstance(value, (np.ndarray, list, tuple)):
                fem_constants[name] = df.fem.Constant(domain, np.asarray(value, dtype=PETSc.ScalarType))
            else:
                fem_constants[name] = df.fem.Constant(domain, PETSc.ScalarType(value))
            
    return fem_states, fem_constants




def create_dirichlet_bcs(V, bc_definitions):
    """
    Creates a list of DirichletBC objects from a list of definitions.
    Handles scalar, vector, and mixed (subspace) problems.
    """
    tdim = V.mesh.topology.dim
    bcs = []
    for bc_def in bc_definitions:
        where_func = bc_def["where"]
        value = bc_def["value"]
        
        facets = df.mesh.locate_entities_boundary(V.mesh, tdim - 1, where_func)
        
        if "subspace_idx" in bc_def:
            # --- Subspace BC Logic ---
            subspace_idx = bc_def["subspace_idx"]
            V_sub, _ = V.sub(subspace_idx).collapse()
            # Locate dofs on the parent space's subspace
            dofs = df.fem.locate_dofs_topological((V.sub(subspace_idx), V_sub), tdim - 1, facets)
            # Create the BC value function on the collapsed subspace
            uD = df.fem.Function(V_sub)
            # Create the BC object targeting the original subspace
            bc_obj = df.fem.dirichletbc(uD, dofs, V.sub(subspace_idx))
        else:
            # --- Standard BC Logic ---
            V_sub = V
            dofs = df.fem.locate_dofs_topological(V_sub, tdim - 1, facets)
            uD = df.fem.Function(V_sub)
            bc_obj = df.fem.dirichletbc(uD, dofs)

        # Interpolate the value (common for both cases)
        if callable(value):
            uD.interpolate(value)
        else:
            if isinstance(value, (list, tuple, np.ndarray)):
                # General way to handle vector values for any dimension
                const_vals = np.array(value, dtype=PETSc.ScalarType)
                uD.interpolate(lambda x: np.outer(const_vals, np.ones(x.shape[1])))
            else:
                ###
                uD.interpolate(lambda x: np.full(x.shape[1], value))

        bcs.append(bc_obj)
        
    return bcs

def create_solver(domain, a_form, L_form, bcs, problem_type="linear", uh=None, J_form=None):
    """
    Creates a solver for a linear or non-linear problem.
    """
    if problem_type == "linear":
        # This branch is for Heat and Picard-linearized Richards
        problem = petsc.LinearProblem(a_form, L_form, bcs=bcs, u=uh,
                                      petsc_options={"ksp_type": "cg", "pc_type": "hypre"})
        return problem.solve
         
    elif problem_type == "nonlinear":
        if uh is None:
            raise ValueError("For nonlinear problems, you must provide the solution function 'uh'.")
        F = a_form
        J = J_form
        problem = petsc.NonlinearProblem(F, uh, bcs=bcs, J=J)
        
        solver = nls.petsc.NewtonSolver(domain.comm, problem)
        solver.relaxation_parameter = 0.8
        solver.convergence_criterion = "incremental"
        solver.max_it = 50
        
        # --- ROBUST SOLVER SETTINGS ---
        solver.ksp_type = "gmres"
        solver.pc_type = "hypre"
        solver.pc_hypre_type = "boomeramg"
        solver.snes_lag_jacobian = -1
        #solver.ksp_type = "preonly"
        #solver.pc_type = "lu"
        return solver.solve
    elif problem_type == 'high_nonlinear': 
        if uh is None:
            raise ValueError("For nonlinear problems, you must provide the solution function 'uh'.")
        F = a_form
        J = J_form
        problem = petsc.NonlinearProblem(F, uh, bcs=bcs, J=J)
        
        solver = nls.petsc.NewtonSolver(domain.comm, problem)
        
        solver.convergence_criterion = "incremental"
        solver.max_it = 50
        solver.relaxation_parameter = 0.9
        
        #  iterative solver (GMRES) with a strong preconditioner (AMG)
        ksp = solver.krylov_solver
        ksp.setType(PETSc.KSP.Type.GMRES)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.HYPRE)
        pc.setHYPREType("boomeramg")
        
        return solver.solve

    
def get_dt(comm, evaluation_times):
    if evaluation_times is not None and len(evaluation_times) > 1:
        min_eval_interval = np.min(np.diff(np.sort(np.unique(evaluation_times))))
        suggested_dt = min_eval_interval / 100 
        dt_fem_internal = max(0.01, min(suggested_dt, 1.0))
    else:
        dt_fem_internal = 0.1 

    if comm.rank == 0:
        print(f"Using initial FEM time step: {dt_fem_internal}")
    return dt_fem_internal