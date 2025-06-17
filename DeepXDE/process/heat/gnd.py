import os
import dolfinx as df
import numpy as np
import ufl
from petsc4py import PETSc
from mpi4py import MPI

from dolfinx.fem.petsc import LinearProblem

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

    ## oder
    #element = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1) 
    #V = df.fem.functionspace(mesh, element)

    left_value = 100.0
    right_value = 0.0

    left_dofs = df.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], x_min))
    right_dofs = df.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], x_max))

    bc_left = df.fem.dirichletbc(PETSc.ScalarType(left_value), left_dofs, V)
    bc_right = df.fem.dirichletbc(PETSc.ScalarType(right_value), right_dofs, V)
    bcs = [bc_left, bc_right]

    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Bilinear and linear forms
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = df.fem.Constant(mesh, PETSc.ScalarType(0.0)) * v * ufl.dx  # mybe df.fem.dx

    # Create the liinear problem
    problem = LinearProblem(a, L, bcs=bcs,
                                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    # Solve
    u_sol = problem.solve()

    with df.io.VTKFile(MPI.COMM_WORLD, "BASELINE/heat/pyTest.pvd", "w") as vtkfile:
        vtkfile.write_function(u_sol)

    # Evaluate solution at a point
    #center_point = np.array([[0.5, 0.5, 0.0]])  # DOLFINx expects 3D coordinates
    #u_values = u_sol.eval(center_point, mesh.)
    #print(f"Temperature at center: {u_values[0][0]}")

    return u_sol


def get_transient_fem(domain_vars):

    x_min, x_max = domain_vars.spatial['x']
    y_min, y_max = domain_vars.spatial['y']
    t_min, t_max = domain_vars.temporal['t']

    dt_val = 100 # Time step size


    mesh = df.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([x_min, y_min]), np.array([x_max, y_max])],
        [100, 100],
        df.mesh.CellType.triangle
    )
    V = df.fem.functionspace(mesh, ("Lagrange", 1))

    u_n = df.fem.Function(V)
    u_n.name = "T_previous"
    u_n.interpolate(lambda x: np.full(x.shape[1], 0.0))

    u_sol = df.fem.Function(V)
    u_sol.name = "T" # Name for VTK output
    u_sol.interpolate(u_n) # Initialize u_sol with the initial condition

    # 3. Define boundary conditions
    left_dofs = df.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], x_min))
    right_dofs = df.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], x_max))

    bc_left = df.fem.dirichletbc(PETSc.ScalarType(100), left_dofs, V)
    bc_right = df.fem.dirichletbc(PETSc.ScalarType(0), right_dofs, V)
    bcs = [bc_left, bc_right]

    # 4. Define variational problem (Backward Euler)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Constant for thermal diffusivity
    alpha = df.fem.Constant(mesh, PETSc.ScalarType(materialData.alpha_thermal_diffusivity))

    
    a_form = u * v * ufl.dx + dt_val * alpha * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = u_n * v * ufl.dx

    problem = LinearProblem(a_form, L_form, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    output_dir = os.path.dirname('BASELINE/heat/transient/test_fem_solution.pvd')
    if MPI.COMM_WORLD.rank == 0:
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    MPI.COMM_WORLD.barrier() # Ensure directory is created by all processes

    vtkfile = df.io.VTKFile(MPI.COMM_WORLD, 'BASELINE/heat/transient/test_fem_solution.pvd', "w")
    vtkfile.write_function(u_sol, t_min) # Save initial state

    # 6. Time stepping loop
    t_current = t_min
    num_steps = int(round((t_max - t_min) / dt_val))

    for i in range(num_steps):
        t_current += dt_val

        # u_n already holds the solution from the previous time step.
        # The LinearProblem is defined using u_n for the RHS.
        
        temp_solution_at_step = problem.solve()
        u_sol.x.array[:] = temp_solution_at_step.x.array
        
        u_n.x.array[:] = u_sol.x.array
        
        vtkfile.write_function(u_sol, t_current)

        #if MPI.COMM_WORLD.rank == 0:
        #    print(f"Time step {i+1}/{num_steps}, t = {t_current:.4f} s, Min T: {u_sol.x.array.min():.2f}, Max T: {u_sol.x.array.max():.2f}")
        
        if t_current >= t_max - 1e-6: # Epsilon for float comparison
            break

    vtkfile.close()
    
    if MPI.COMM_WORLD.rank == 0:
        print(f"Transient FEM solution completed. Output saved to BASELINE/heat/transient/test_fem_solution.pvd")

    return u_sol # Return the solution at the final time step
