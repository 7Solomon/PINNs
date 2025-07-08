import numpy as np
from utils.metadata import Domain
from typing import Dict, Any, Tuple, Optional

def get_evaluation_points(domain: Domain, 
                evaluation_times: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Generate all point formats needed for FEM, visualization, and model prediction.
    
    Args:
        domain: Domain object with spatial/temporal bounds
        evaluation_times: Optional array of times for transient problems
    
    Returns:
        Dictionary with all point arrays in different formats
    """
    result = {}
    
    # Extract spatial bounds and create coordinate arrays
    spatial_dims = list(domain.spatial.keys())
    spatial_coords = {}
    
    for dim in spatial_dims:
        min_val, max_val = domain.spatial[dim]
        if dim not in domain.resolution:
            raise ValueError(f"Resolution must contain key '{dim}'")
        spatial_coords[dim] = np.linspace(min_val, max_val, domain.resolution[dim])

    result['spatial_coords'] = spatial_coords
    result['spatial_dims'] = spatial_dims
    result['resolution'] = domain.resolution
    
    # === SPATIAL MESHGRIDS ===
    if len(spatial_dims) == 2:
        x_key, y_key = spatial_dims
        nx, ny = domain.resolution[x_key], domain.resolution[y_key]
        
        # Create both indexing types for different use cases
        X_ij, Y_ij = np.meshgrid(spatial_coords[x_key], spatial_coords[y_key], indexing='ij')
        X_xy, Y_xy = np.meshgrid(spatial_coords[x_key], spatial_coords[y_key], indexing='xy')
        
        result['spatial_meshgrid'] = {
            'ij': {x_key: X_ij, y_key: Y_ij},  # (nx, ny) - for FEM consistency
            'xy': {x_key: X_xy, y_key: Y_xy},  # (ny, nx) - for matplotlib
        }
        
        # Flattened points for FEM evaluation - always use 'ij' for consistency
        result['spatial_points_flat'] = np.stack([X_ij.ravel(), Y_ij.ravel()], axis=-1)
        result['reshape_static_to_grid'] = lambda data: _reshape_static_to_grid(data, nx, ny)

    elif len(spatial_dims) == 1:
        x_key = spatial_dims[0]
        nx = domain.resolution[x_key]
        result['spatial_meshgrid'] = {x_key: spatial_coords[x_key]}
        result['spatial_points_flat'] = spatial_coords[x_key][:, np.newaxis]
    
    # === TEMPORAL HANDLING ===
    if domain.temporal:
        t_key = list(domain.temporal.keys())[0]
        t_min, t_max = domain.temporal[t_key]
        
        if evaluation_times is not None:
            t_coords = evaluation_times
        else:
            t_coords = np.linspace(t_min, t_max, domain.resolution[t_key])

        result['temporal_coords'] = {t_key: t_coords}
        result['temporal_key'] = t_key
        
        # === SPACETIME MESHGRIDS ===
        if len(spatial_dims) == 2:
            x_key, y_key = spatial_dims
            nt = len(t_coords)
            
            # Create spacetime meshgrids for different use cases
            X_ij, Y_ij, T_ij = np.meshgrid(spatial_coords[x_key], spatial_coords[y_key], t_coords, indexing='ij')
            X_xy, Y_xy, T_xy = np.meshgrid(spatial_coords[x_key], spatial_coords[y_key], t_coords, indexing='xy')
            
            result['spacetime_meshgrid'] = {
                'ij': {x_key: X_ij, y_key: Y_ij, t_key: T_ij},  # (nx, ny, nt)
                'xy': {x_key: X_xy, y_key: Y_xy, t_key: T_xy},  # (ny, nx, nt)
            }
            # Flattened spacetime points for model prediction - use 'ij' for consistency
            result['spacetime_points_flat'] = np.vstack([X_ij.ravel(), Y_ij.ravel(), T_ij.ravel()]).T

            # === RESHAPING UTILITIES ===
            result['reshape_utils'] = {
                'fem_to_ij': lambda data: _reshape_fem_to_grid(data, nx, ny, nt, 'ij'),
                'fem_to_xy': lambda data: _reshape_fem_to_grid(data, nx, ny, nt, 'xy'),
                'ij_to_xy': lambda data: np.transpose(data, (1, 0, 2)),
                'xy_to_ij': lambda data: np.transpose(data, (1, 0, 2)),
                'pred_to_ij': lambda data: _pred_to_ij(data, nx, ny, nt),
            }
        
        elif len(spatial_dims) == 1:
            x_key = spatial_dims[0]
            nt = len(t_coords)
            X, T = np.meshgrid(spatial_coords[x_key], t_coords, indexing='ij')
            result['spacetime_meshgrid'] = {x_key: X, t_key: T}
            result['spacetime_points_flat'] = np.vstack([X.ravel(), T.ravel()]).T

            result['reshape_utils'] = {
                'fem_to_ij': lambda data: data.T.reshape(nx, nt) ,  # weird because 1d no ij but it reshpaes to  (nz, nt)
                'pred_to_ij': lambda data: _pred_to_ij(data, nx, None, nt),
                }

    
    return result


def _pred_to_ij(predictions, nx, ny, nt):
    """
    Reshape predictions from (nt, nx*ny) to (nx, ny, nt) for 'ij' indexing.
    
    Args:
        predictions: Array of shape (nt, nx*ny) or (nt, nx*ny, n_components)
        nx, ny, nt: Grid dimensions
    
    Returns:
        Reshaped array in (nx, ny, nt) format
    """
    if ny is None:
        # 1D case, reshape to (nx, nt)
        if predictions.ndim == 1:
            return predictions.reshape(nx, nt).transpose(1, 0)
        elif predictions.ndim == 2:
            reshaped = predictions.reshape(nt, nx, -1).transpose(1, 0, 2)
            if reshaped.shape[-1] == 1:
                return reshaped.squeeze(-1)
        else:
            raise ValueError(f"Unexpected predictions shape: {predictions.shape}")
    else:
        if predictions.ndim == 2:
            n_components = predictions.shape[1]
            reshaped = predictions.reshape(nt, nx, ny, n_components).transpose(1, 2, 0, 3)
            if n_components == 1:
                return reshaped.squeeze(-1)
            return reshaped
        #elif predictions.ndim == 3:
        #    reshaped = predictions.reshape(nt, nx, ny, -1).transpose(1, 2, 0, 3)
        #    if reshaped.shape[-1] == 1:
        #        return reshaped.squeeze(-1)
        #    return reshaped
        else:
            raise ValueError(f"Unexpected predictions shape: {predictions.shape}")
def _reshape_static_to_grid(static_data, nx, ny):
    """
    Reshapes flat static data (e.g., from FEM evaluation) into a grid.
    """
    n_points = nx * ny
    if static_data.shape[0] != n_points:
        raise ValueError(f"Data size {static_data.shape[0]} does not match grid dimensions {nx}x{ny}={n_points}")

    if static_data.ndim == 1:
        return static_data.reshape(nx, ny)
    else:
        n_components = static_data.shape[1]
        return static_data.reshape(nx, ny, n_components)


def _reshape_fem_to_grid(fem_data, nx, ny, nt, indexing='ij'):
    """
    Reshape FEM evaluation data to grid format.
    
    Args:
        fem_data: Array from execute_transient_simulation with shape (nt, nx*ny) or (nt, nx*ny, n_components)
        nx, ny, nt: Grid dimensions
        indexing: 'ij' for (nx, ny, nt) or 'xy' for (ny, nx, nt)
    """
    # Handle different input shapes
    if fem_data.ndim == 3 and fem_data.shape[-1] == 1:
        fem_data = fem_data.squeeze(-1)  # Remove singleton component dimension
    
    if fem_data.shape == (nt, nx * ny):
        # Standard case: reshape from (nt, nx*ny) to (nx, ny, nt)
        reshaped = fem_data.T.reshape(nx, ny, nt)
    elif fem_data.ndim == 3:
        # Handle case with multiple components, e.g., (nt, nx*ny, n_components)
        reshaped = fem_data.reshape(nt, nx, ny, -1).transpose(1, 2, 0, 3)
    else:
        raise ValueError(f"Unexpected fem_data shape: {fem_data.shape}")
    
    if indexing == 'xy':
        # Transpose to (ny, nx, nt) for matplotlib
        reshaped = np.transpose(reshaped, (1, 0, 2))
    
    return reshaped

def get_meshgrid_for_visualization(points_data, indexing='xy'):
    """Get meshgrid in the format expected by matplotlib."""
    if 'spacetime_meshgrid' in points_data:
        return points_data['spacetime_meshgrid'][indexing]
    else:
        return points_data['spatial_meshgrid'][indexing]

def get_meshgrid_for_fem(points_data):
    """Get meshgrid in the format consistent with FEM evaluation."""
    if 'spacetime_meshgrid' in points_data:
        return points_data['spacetime_meshgrid']['ij']
    else:
        return points_data['spatial_meshgrid']['ij']