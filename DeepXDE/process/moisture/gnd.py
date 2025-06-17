import fenicsx as fe
def get_1d_head_fem(domain_vars):
    z_min, z_max = domain_vars.spatial['z']

    mesh = fe.IntervalMesh(32, z_min, z_max)

    V = fe.FunctionSpace(mesh, 'P', 1)

    # BCs
    left_value = -1.0
    right_value = -7.0
    initial_value = -3.5

    u_left = fe.Constant(left_value)
    u_right = fe.Constant(right_value)
    u_initial = fe.Constant(initial_value)

    bc_left = fe.DirichletBC(V, u_left, lambda x: fe.near(x[0], z_min))
    bc_right = fe.DirichletBC(V, u_right, lambda x: fe.near(x[0], z_max))
    bc_initial = fe.InitialCondition(V, u_initial, lambda x: fe.near(x[0], z_min))
    bcs = [bc_left, bc_right, bc_initial]

    # V problem
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)