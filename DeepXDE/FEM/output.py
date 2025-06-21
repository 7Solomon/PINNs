import os
import numpy as np
from dolfinx import geometry
from mpi4py import MPI
import dolfinx as df

def prepare_output_directory(output_dir_path, comm):
    if comm.rank == 0:
        output_dir = os.path.dirname(output_dir_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    comm.barrier()

def initialize_point_evaluation(domain, evaluation_spatial_points, comm):
    perform_point_evaluation = evaluation_spatial_points is not None
    eval_points_3d = None
    bb_tree = None

    if perform_point_evaluation:
        if evaluation_spatial_points.ndim != 2:
            raise ValueError("evaluation_spatial_points must be a 2D array.")
        
        num_points = evaluation_spatial_points.shape[0]
        num_cols = evaluation_spatial_points.shape[1]

        # Generalize point padding for 1D, 2D, or 3D input
        if num_cols == 1: # 1D case (z) -> pad to (z, 0, 0)
            eval_points_3d = np.zeros((num_points, 3), dtype=np.float64)
            eval_points_3d[:, 0] = evaluation_spatial_points[:, 0]
        elif num_cols == 2: # 2D case (x, y) -> pad to (x, y, 0)
            eval_points_3d = np.zeros((num_points, 3), dtype=np.float64)
            eval_points_3d[:, :2] = evaluation_spatial_points
        elif num_cols == 3: # 3D case
            eval_points_3d = evaluation_spatial_points.astype(np.float64)
        else:
            raise ValueError(f"evaluation_spatial_points must have 1, 2, or 3 columns, but got {num_cols}.")

        eval_points_3d = np.ascontiguousarray(eval_points_3d)
        
        try:
            bb_tree = geometry.BoundingBoxTree(domain, domain.topology.dim)
        except TypeError: # Try older API
            try:
                bb_tree = geometry.bb_tree(domain, domain.topology.dim)
            except (TypeError, AttributeError) as e:
                if comm.rank == 0:
                    print(f"Error creating BoundingBoxTree: {e}")
                    print("Available geometry functions:", [attr for attr in dir(geometry) if 'tree' in attr.lower() or 'bound' in attr.lower()])
                raise RuntimeError("Could not create BoundingBoxTree with available API") from e
                
    return perform_point_evaluation, eval_points_3d, bb_tree

def evaluate_solution_at_points_on_rank_0(uh, eval_points_3d, bb_tree, domain, comm):
    output_dim = uh.function_space.dofmap.bs

    local_evals_at_t = []
    if eval_points_3d is not None and bb_tree is not None:
        cell_candidates = geometry.compute_collisions_points(bb_tree, eval_points_3d)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, eval_points_3d)
        
        for i in range(eval_points_3d.shape[0]):
            if len(colliding_cells.links(i)) > 0:
                local_cell = colliding_cells.links(i)[0]
                try:
                    value = uh.eval(eval_points_3d[i:i+1], np.array([local_cell], dtype=np.int32))
                    local_evals_at_t.append((i, value[0]))
                except Exception as e:
                    if comm.rank == 0:
                        print(f"Warning: Failed to evaluate point {i} at current time: {e}")
    
    gathered_evals = comm.gather(local_evals_at_t, root=0)
    
    if comm.rank == 0:
        if output_dim == 1:
            results_for_t = np.full(eval_points_3d.shape[0], np.nan)
        else:
            results_for_t = np.full((eval_points_3d.shape[0], output_dim), np.nan)

        if gathered_evals:
            for rank_data in gathered_evals:
                if rank_data:
                    for global_idx, val in rank_data:
                        results_for_t[global_idx] = val
        return results_for_t
    return None


def save_fem_results(filepath: str, data: np.ndarray):
    """
    Saves the FEM results (numpy array) to a file.

    This function should be called from all MPI processes, but only the root
    process (rank 0) will perform the save operation.

    Args:
        filepath (str): The path to the file where results will be saved.
        data (np.ndarray): The numpy array containing the data. On ranks other
                           than 0, this can be None, but the root process must
                           have the complete data.
    """
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        if data is not None:
            output_dir = os.path.dirname(filepath)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            np.save(filepath, data)
            print(f"FEM results saved to {filepath}")
        else:
            print("Warning (rank 0): No data provided to save.")

def load_fem_results(filepath: str):
    """
    Loads FEM results from a file.

    In an MPI environment, only the root process (rank 0) loads the data.

    Args:
        filepath (str): The path to the file from which to load results.

    Returns:
        np.ndarray or None: The loaded numpy array on the root process (rank 0)
                            if the file exists, otherwise None. On all other
                            processes, this function returns None.
    """
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        if os.path.exists(filepath):
            try:
                print(f"Loading FEM results from {filepath}")
                return np.load(filepath)
            except Exception as e:
                print(f"Error loading FEM results from {filepath}: {e}")
                return None
        else:
            # File not found is a normal case for the first run, so no warning.
            return None
    return None