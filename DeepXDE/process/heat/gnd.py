import os
import dolfinx as df
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem.petsc import LinearProblem

from FEM.init_helper import create_dirichlet_bcs, create_mesh_and_function_space, create_solver, get_dt, initialize_fem_state
from FEM.run_helper import execute_transient_simulation
from material import concreteData

materialData = concreteData


def get_steady_fem(domain_vars):
        
    x_min, x_max = domain_vars.spatial['x']
    y_min, y_max = domain_vars.spatial['y']

    mesh = df.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[x_min, y_min], [x_max, y_max]],
        [10, 10], 
        df.mesh.CellType.triangle
    )


    V = df.fem.functionspace(mesh, ("Lagrange", 1))

    left_value = 100.0
    right_value = 0.0

    left_dofs = df.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], x_min))
    right_dofs = df.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], x_max))

    bc_left = df.fem.dirichletbc(PETSc.ScalarType(left_value), left_dofs, V)
    bc_right = df.fem.dirichletbc(PETSc.ScalarType(right_value), right_dofs, V)
    bcs = [bc_left, bc_right]

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = df.fem.Constant(mesh, PETSc.ScalarType(0.0)) * v * ufl.dx

    problem = LinearProblem(a, L, bcs=bcs,
                                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    u_sol = problem.solve()

    output_dir_steady = os.path.dirname("BASELINE/heat/pyTest.pvd")
    if MPI.COMM_WORLD.rank == 0:
        if output_dir_steady and not os.path.exists(output_dir_steady):
            os.makedirs(output_dir_steady)
    MPI.COMM_WORLD.barrier()

    #with df.io.VTKFile(MPI.COMM_WORLD, "BASELINE/heat/pyTest.pvd", "w") as vtkfile:
    #    vtkfile.write_function(u_sol)

    return u_sol

# --- Helper functions for get_transient_fem ---

#def _initialize_fem_state(domain, V, alpha_value, dt_value, initial_temp_value=0.0):
#    alpha = df.fem.Constant(domain, PETSc.ScalarType(alpha_value))
#    dt_const = df.fem.Constant(domain, PETSc.ScalarType(dt_value))
#
#    un = df.fem.Function(V)
#    un.name = "u_previous"
#    un.interpolate(lambda x: np.full(x.shape[1], initial_temp_value))
#
#    uh = df.fem.Function(V)
#    uh.name = "temperature"
#    uh.interpolate(un)
#    return alpha, dt_const, un, uh

def define_heat_equation_forms(V, dt, alpha, un):
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = u * v * ufl.dx + dt * alpha * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = un * v * ufl.dx
    return df.fem.form(a), df.fem.form(L)

#def _setup_constant_dirichlet_bcs(domain, V, x_min, x_max, left_value, right_value):
#    # Right boundary (x = x_max) at right_value
#    def dirichlet_boundary_at_x_max(x):
#        return np.isclose(x[0], x_max)
#    
#    tdim = domain.topology.dim
#    bc_facets_right = df.mesh.locate_entities_boundary(domain, tdim - 1, dirichlet_boundary_at_x_max)
#    bndry_dofs_right = df.fem.locate_dofs_topological(V, tdim - 1, bc_facets_right)
#    uD_right = df.fem.Function(V)
#    uD_right.interpolate(lambda x: np.full(x.shape[1], right_value))
#    bc_right = df.fem.dirichletbc(uD_right, bndry_dofs_right)
#
#    # Left boundary (x = x_min) at left_value
#    def dirichlet_boundary_at_x_min(x):
#        return np.isclose(x[0], x_min)
#
#    bc_facets_left = df.mesh.locate_entities_boundary(domain, tdim - 1, dirichlet_boundary_at_x_min)
#    bndry_dofs_left = df.fem.locate_dofs_topological(V, tdim - 1, bc_facets_left)
#    uD_left = df.fem.Function(V)
#    uD_left.interpolate(lambda x: np.full(x.shape[1], left_value))
#    bc_left = df.fem.dirichletbc(uD_left, bndry_dofs_left)
#    
#    return [bc_left, bc_right]

#def _compile_forms_and_assemble_matrix(domain, a_form, L_form_template, un, bcs):
#    compiled_a = df.fem.form(a_form)
#    A = petsc.assemble_matrix(compiled_a, bcs=bcs)
#    A.assemble()
#
#    # L_form depends on 'un', so compile it here or pass 'un' to assembly
#    compiled_L = df.fem.form(L_form_template(un))
#    b_vec = petsc.create_vector(compiled_L)
#    return A, b_vec, compiled_a, compiled_L


#def _create_ksp_solver(domain, A):
#    solver = PETSc.KSP().create(domain.comm)
#    solver.setOperators(A)
#    solver.setType(PETSc.KSP.Type.CG)
#    pc = solver.getPC()
#    pc.setType(PETSc.PC.Type.HYPRE)
#    pc.setHYPREType("boomeramg")
#    return solver


def get_transient_fem(domain_vars, 
                        grid_resolution: tuple = (80, 60),
                        evaluation_times: np.ndarray = None, 
                        evaluation_spatial_points_xy: np.ndarray = None):


    # DEF VARS
    comm = MPI.COMM_WORLD
    rank = comm.rank
    x_min, x_max = domain_vars.spatial['x']
    y_min, y_max = domain_vars.spatial['y']
    t_min, t_max = domain_vars.temporal['t']
    Nx, Ny = grid_resolution

    # GET dt
    dt_fem_internal = get_dt(comm, evaluation_times)

    ### MESH
    element_description =  {"type": "scalar", "family": "Lagrange", "degree": 1}
    domain, V = create_mesh_and_function_space(comm, [[x_min, y_min], [x_max, y_max]], [Nx, Ny], element_description)

    initial_conditions = {'temperature': 0.0}
    constants_def = {"dt": dt_fem_internal, "alpha": materialData.alpha_thermal_diffusivity}
    state_vars = ["uh_temperature", "un_temperature"]
    fem_states, fem_constants = initialize_fem_state(
        V,
        initial_conditions=initial_conditions,
        element_desc=element_description,
        constants_def=constants_def,
        state_vars=state_vars
    )
    # Extract
    uh = fem_states["uh_temperature"]
    un = fem_states["un_temperature"]
    dt_const = fem_constants["dt"]
    alpha_const = fem_constants["alpha"]
    
    # Set the initial condition for the previous step as well
    #un.x.array[:] = uh.x.array

    a_form, l_form = define_heat_equation_forms(V, dt_const, alpha_const, un)

    bc_left_value = 100.0
    bc_right_value = 0.0
    bcs = [
            {"where": lambda x: np.isclose(x[0], x_min), "value": bc_left_value, "component": None},
            {"where": lambda x: np.isclose(x[0], x_max), "value": bc_right_value, "component": None}
        ]
    bcs = create_dirichlet_bcs(V, bcs)

    #A, b_vec, compiled_a, compiled_L = _compile_forms_and_assemble_matrix(domain, a_form, L_form_template, un, bcs)
    #solver = _create_ksp_solver(domain, A)
    solver_function = create_solver(domain, a_form, l_form, bcs, 'linear', uh=uh)

    uh, final_evaluated_data = execute_transient_simulation(
        domain=domain,
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

