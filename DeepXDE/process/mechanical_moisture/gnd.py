import fenics as fe
from material import concreteData, sandData

materialData = concreteData
    
def strain(u):
    return 0.5 * (fe.grad(u) + fe.grad(u).T)
def stress(u):
    lambda_ = materialData.E * materialData.nu / ((1 + materialData.nu) * (1 - materialData.nu))
    mu = materialData.E / (2 * (1 + materialData.nu))
    return lambda_ * fe.tr(strain(u)) * fe.Identity(len(u)) + 2 * mu * strain(u)



def get_2d_fem_solution(domain_vars):
    x_min, x_max = domain_vars.spatial['x']
    y_min, y_max = domain_vars.spatial['y']

    mesh = fe.RectangleMesh(fe.Point(x_min, y_min), fe.Point(x_max, y_max), 64, 32)

    V = fe.VectorFunctionSpace(mesh, 'P', 1)

    # BCs
    clamped_boundary = f"on_boundary && near(x[0], {x_min})"
    bc = fe.DirichletBC(V, fe.Constant((0.0, 0.0)), clamped_boundary)

    f_body = fe.Constant((1.0, 0.0))

    # WF
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)

    # integral(sigma(u) : epsilon(v)) * dx = integral(f_body . v) * dx
    equilibrium = fe.inner(stress(u), strain(v)) * fe.dx
    body_force = fe.dot(f_body, v) * fe.dx

    # SOL
    u_sol = fe.Function(V, name="Displacement")
    fe.solve(equilibrium == body_force, u_sol, bc)

    return u_sol