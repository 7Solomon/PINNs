import math
import numpy as np
import dolfinx as df
import ufl
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.fem.petsc import LinearProblem
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
def stress(u):
    lambda_ = materialData.E * materialData.nu / ((1 + materialData.nu) * (1 - materialData.nu))
    mu = materialData.E / (2 * (1 + materialData.nu))
    return lambda_ * ufl.tr(strain(u)) * ufl.Identity(len(u)) + 2 * mu * strain(u)

def clamped_boundary_condition(x, x_min):
    return np.isclose(x[0], x_min)
    
def get_einspannung_2d_fem(domain_vars):
    x_min, x_max = domain_vars.spatial['x']
    y_min, y_max = domain_vars.spatial['y']

    mesh = df.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[x_min, y_min], [x_max, y_max]],
        [10, 10], 
        df.mesh.CellType.triangle
    )

    # For 2D vector problem (u,v displacement) this is VectorFunctionSpace
    V = df.fem.functionspace(mesh, ('P', 1, (2,)))
    
    # BCs
    dofs = df.fem.locate_dofs_geometrical(V, lambda x: clamped_boundary_condition(x, x_min))
    bc = df.fem.dirichletbc(np.array([0.0, 0.0]), dofs, V)

    f_body = df.fem.Constant(mesh, (1.0, 0.0))


    # WF
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # integral(sigma(u) : epsilon(v)) * dx = integral(f_body . v) * dx
    a = ufl.inner(stress(u), strain(v)) * ufl.dx
    L = ufl.dot(f_body, v) * ufl.dx

    # SOL
    #u_sol = df.fem.Function(V, name="Displacement")
    problem = LinearProblem(a, L, bcs=[bc],
                                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    u_sol = problem.solve()
    
    
    #print(f"Solution function space value size: {u_sol.function_space.element.value_size}")
    
    #ufl.solve(a == L, u_sol, bc)

    #with df.io.VTKFile(MPI.COMM_WORLD, "displacement_fem_test.pvd", "w") as vtk:
    #    vtk.write_function(u_sol)
    return u_sol


