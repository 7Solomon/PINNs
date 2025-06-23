import math
import numpy as np
import dolfinx as df
import ufl
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.fem.petsc import LinearProblem
from FEM.init_helper import create_dirichlet_bcs, create_mesh_and_function_space, create_solver
from material import concreteData, sandData

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
     
def get_einspannung_2d_fem(domain_vars,
                           grid_resolution=(10, 10),
                           ):
    """
    Calculates the displacement of a 2D clamped beam using FEM helpers.
    """
    comm = MPI.COMM_WORLD
    x_min, x_max = domain_vars.spatial['x']
    y_min, y_max = domain_vars.spatial['y']
    nx, ny = grid_resolution

    # MESH
    element_desc = {"family": "Lagrange", "degree": 1, "type": "vector"}
    mesh, V = create_mesh_and_function_space(
        comm,
        domain_extents=[[x_min, y_min], [x_max, y_max]],
        domain_resolution=[nx, ny],
        element_desc=element_desc
    )

    # BC
    bc_definitions = [
        {"where": lambda x: np.isclose(x[0], x_min), "value": (0.0, 0.0)}
    ]
    bcs = create_dirichlet_bcs(V, bc_definitions)

    #  MABYE DIFFRENT VALS
    #f_body = df.fem.Constant(mesh, PETSc.ScalarType((0, -materialData.rho*materialData.g)))
    C = materialData.C_stiffness_matrix().cpu().numpy()
    C = df.fem.Constant(mesh, np.asarray(C, dtype=PETSc.ScalarType))

    a, L = get_einspannung_weak_form(V, C)

    # SOLVER
    solver_function = create_solver(mesh, a, L, bcs, problem_type="linear")
    u_sol = solver_function()
    u_sol.name = "Displacement"
    
    return u_sol