import dolfinx as df
import numpy as np
from dolfinx.fem import Function
import ufl

def evaluate_fem_at_points(fem_solution: Function, points: np.ndarray):
    """
    Evaluate DOLFiNx function at given points with robust dimension detection
    """
    mesh = fem_solution.function_space.mesh
    bb_tree = df.geometry.bb_tree(mesh, mesh.topology.dim)
    
    # Use UFL to detect if this is a vector function
    u_test = ufl.TrialFunction(fem_solution.function_space)
    is_vector = len(u_test.ufl_shape) > 0
    
    if is_vector:
        function_dim = u_test.ufl_shape[0]  # Number of components
        #print(f"DEBUG: Vector function detected with {function_dim} components")
    else:
        function_dim = 1
        #print(f"DEBUG: Scalar function detected")
    
    spatial_dim = points.shape[1]
    n_points = points.shape[0]
    
    # Initialize results array
    results = np.zeros((n_points, function_dim))
    
    for i, point in enumerate(points):
        try:
            # Convert to 3D point (DOLFiNx requirement)
            if spatial_dim == 2:
                point_3d = np.array([point[0], point[1], 0.0], dtype=np.float64)
            elif spatial_dim == 1:
                point_3d = np.array([point[0], 0.0, 0.0], dtype=np.float64)
            else:
                point_3d = point.astype(np.float64)
            
            point_3d = point_3d.reshape(1, -1)
            
            # Find cell containing this point
            cell_candidates = df.geometry.compute_collisions_points(bb_tree, point_3d)
            colliding_cells = df.geometry.compute_colliding_cells(mesh, cell_candidates, point_3d)
            
            if len(colliding_cells.array) > 0:
                cell = colliding_cells.array[0]
                value = fem_solution.eval(point_3d, np.array([cell]))
                
                if i == 0:  # Debug first point only
                    print(f"DEBUG: First evaluation - value shape: {value.shape}, value: {value}")
                
                # Handle different return shapes robustly
                value_flat = np.atleast_1d(value.flatten())
                n_components = min(len(value_flat), function_dim)
                results[i, :n_components] = value_flat[:n_components]
                    
        except Exception as e:
            if i == 0:  # Only print error for first point to avoid spam
                print(f"Warning: Failed to evaluate point {i}: {e}")
            results[i, :] = 0.0
    
    print(f"DEBUG: Final results shape: {results.shape}")
    return results