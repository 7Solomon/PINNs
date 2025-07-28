import torch
import pandas as pd
import numpy as np
from scipy.interpolate import interpn

from utils.metadata import Domain

import pyvista as pv
import re


def load_comsol_data_mechanic_2d(filepath):
    """Load COMSOL data from text file"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip().startswith('%') and line.strip():
                # Split by whitespace and convert to float
                values = [float(x) for x in line.strip().split()]
                if len(values) >= 4:  # Ensure we have X, Y, u, v at minimum
                    data.append(values[:4])  # Take first 4 columns: X, Y, u, v
    return np.array(data)


def load_COMSOL_file_data(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            return file_content
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None
    
def analyze_steady_COMSOL_file_data(file_content):
    lines = file_content.strip().split('\n')
    
    header_line_index = -1
    data_start_index = -1
    expected_column_names = ['X', 'Y', 'T (K)']

    for i, line in enumerate(lines):
        if all(name in line for name in expected_column_names):
            header_line_index = i
            data_start_index = i + 1
            break

    if header_line_index == -1:
        raise ValueError(f"Could not find the data header line containing {expected_column_names} in the COMSOL file.")

    column_names = expected_column_names

    data_lines = lines[data_start_index:]
    parsed_data = []
    for line in data_lines:
        values = line.strip().split()
        if len(values) == len(column_names):
            try:
                parsed_data.append([float(v) for v in values])
            except ValueError:
                continue
    df = pd.DataFrame(parsed_data, columns=column_names)
    domain_tensor = torch.tensor(df[['X', 'Y']].values, dtype=torch.float32, device=torch.device('cpu'))
    temp_tensor = torch.tensor(df['T (K)'].values, dtype=torch.float32, device=torch.device('cpu')).unsqueeze(1) - 273.15  # Convert to Celsius

    return domain_tensor, temp_tensor

def test_load_comsol_data(filepath):
    """
    Universal COMSOL data loader that handles both steady-state and time-dependent cases.
    
    Args:
        filepath (str): Path to the COMSOL text file
        
    Returns:
        dict: Dictionary containing loaded data with keys:
            - 'coordinates': numpy array of spatial coordinates [X, Y, ...]
            - 'fields': numpy array of field values
            - 'time_points': numpy array of time values (if time-dependent)
            - 'is_time_dependent': boolean indicating if data is time-dependent
            - 'field_names': list of field variable names
            - 'coordinate_names': list of coordinate variable names
    """
    data_lines = []
    header_info = {}
    field_names = []
    coordinate_names = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('%'):
                # Parse header information
                if 'Dimension:' in line:
                    header_info['dimension'] = int(line.split(':')[1].strip())
                elif 'Nodes:' in line:
                    header_info['nodes'] = int(line.split(':')[1].strip())
                elif 'Expressions:' in line:
                    header_info['expressions'] = int(line.split(':')[1].strip())
                elif 'Description:' in line:
                    description = line.split(':')[1].strip()
                    header_info['description'] = description
                # Parse column headers (the line with X, Y, field names)
                elif any(coord in line for coord in ['X', 'Y', 'Z']) and not line.startswith('% Model'):
                    # This is the header line with column names
                    columns = line[1:].strip().split()  # Remove % and split
                    
                    # Identify coordinates and fields
                    for i, col in enumerate(columns):
                        if col in ['X', 'Y', 'Z']:
                            coordinate_names.append(col)
                        else:
                            # Handle time-dependent field names like "T (K) @ t=3600"
                            if '@' in col:
                                # Extract base field name
                                base_name = col.split('(')[0].strip()
                                if base_name not in field_names:
                                    field_names.append(base_name)
                            else:
                                # Regular field name
                                field_name = col.split('(')[0].strip()  # Remove units
                                if field_name not in coordinate_names and field_name not in field_names:
                                    field_names.append(field_name)
            elif line and not line.startswith('%'):
                # Data line
                values = [float(x) for x in line.split()]
                if len(values) > 0:
                    data_lines.append(values)
    
    if not data_lines:
        raise ValueError("No data found in file")
    
    data_array = np.array(data_lines)
    
    # Determine the structure
    num_coords = len(coordinate_names)
    num_cols = data_array.shape[1]
    
    # Extract coordinates
    coordinates = data_array[:, :num_coords]
    
    # Determine if time-dependent
    is_time_dependent = num_cols > (num_coords + len(field_names))
    
    if is_time_dependent:
        # For time-dependent data, we need to figure out the time structure
        # Count how many time steps we have
        remaining_cols = num_cols - num_coords
        
        # Check if we can determine time points from header
        time_points = []
        with open(filepath, 'r') as f:
            header_line = ""
            for line in f:
                if any(coord in line for coord in ['X', 'Y', 'Z']) and '@' in line:
                    header_line = line
                    break
        
        if '@' in header_line:
            # Extract time points from column headers
            import re
            time_matches = re.findall(r'@ t=([0-9.E]+)', header_line)
            time_points = [float(t) for t in time_matches]
        
        if len(time_points) == 0:
            # Fallback: assume equal time spacing
            num_time_steps = remaining_cols // len(field_names) if len(field_names) > 0 else remaining_cols
            time_points = np.arange(num_time_steps)
        
        # Reshape field data for time-dependent case
        num_time_steps = len(time_points)
        num_field_vars = len(field_names)
        
        if num_field_vars == 0:
            # Single field variable case
            field_data = data_array[:, num_coords:].reshape(-1, num_time_steps)
            field_names = ['field']
        else:
            # Multiple field variables
            field_data = data_array[:, num_coords:].reshape(-1, num_time_steps, num_field_vars)
        
        time_points = np.array(time_points)
    else:
        # Steady-state case
        field_data = data_array[:, num_coords:]
        time_points = None
        
        # If no field names were detected, create generic ones
        if len(field_names) == 0:
            num_fields = field_data.shape[1]
            field_names = [f'field_{i}' for i in range(num_fields)]
    
    return {
        'coordinates': coordinates,
        'fields': field_data,
        'time_points': time_points,
        'is_time_dependent': is_time_dependent,
        'field_names': field_names,
        'coordinate_names': coordinate_names,
        'header_info': header_info
    }


def extract_time_series_data_thermo_mechanical(filename):
    """
    Extract time series data from VTU file and organize into (nx, ny, nt, 3) array
    where the 3 components are [displacement_x, displacement_y, temperature]
    """
    # Loadthe VTU file
    mesh = pv.read(filename)
    
    # Print mesh summary
    #print(mesh)
    #print("Point data:", mesh.point_data.keys())
    #print("Cell data:", mesh.cell_data.keys())
    
    # Get spatial coordinates
    coords = mesh.points  # shape: (n_points, 3)
    n_points = coords.shape[0]
    
    # Extract time steps and their string representations from field names
    time_pattern = r'@_t=([0-9.E+-]+)'
    time_steps_raw = {}  # maps time_value -> original_string
    
    for key in mesh.point_data.keys():
        match = re.search(time_pattern, key)
        if match:
            time_str = match.group(1)
            time_val = float(time_str)
            time_steps_raw[time_val] = time_str
    
    time_steps = sorted(list(time_steps_raw.keys()))
    nt = len(time_steps)
    
    #print(f"Found {nt} time steps: {time_steps}")
    
    # Determine grid dimensions (assuming structured grid)
    # You might need to adjust this based on your specific grid structure
    x_coords = np.unique(coords[:, 0])
    y_coords = np.unique(coords[:, 1])
    nx = len(x_coords)
    ny = len(y_coords)
    
    #print(f"Grid dimensions: nx={nx}, ny={ny}")
    
    # Initialize output array: (nx, ny, nt, 3)
    # 3 components: [displacement_x, displacement_y, temperature]
    data_array = np.zeros((nx, ny, nt, 3))
    
    # Create mapping from coordinates to grid indices
    # This assumes your mesh points are on a regular grid
    coord_to_index = {}
    for i, point in enumerate(coords):
        x_idx = np.argmin(np.abs(x_coords - point[0]))
        y_idx = np.argmin(np.abs(y_coords - point[1]))
        coord_to_index[i] = (x_idx, y_idx)
    
    # Extract data for each time step
    for t_idx, time_val in enumerate(time_steps):
        # Use the original string representation from the file
        time_str = time_steps_raw[time_val]
        search_pattern = f"@_t={time_str}"
        
        #print(f"\n--- Processing time step {t_idx}: {time_val} (using '{time_str}') ---")
        
        # Find field names for this time step
        u_x_key = None
        u_y_key = None
        temp_key = None
        
        for key in mesh.point_data.keys():
            if search_pattern in key:
                if "Displacement_field,_X-component" in key:
                    u_x_key = key
                elif "Displacement_field,_Y-component" in key:
                    u_y_key = key
                elif "Temperature" in key:
                    temp_key = key
        
        #print(f"  Found keys - ux: {u_x_key is not None}, uy: {u_y_key is not None}, T: {temp_key is not None}")
        
        # Extract data for this time step
        if u_x_key:
            u_x_data = mesh.point_data[u_x_key]
            #print(f"  ux data range: [{np.min(u_x_data):.6e}, {np.max(u_x_data):.6e}]")
        else:
            u_x_data = np.zeros(n_points)
            #print(f"  ux data: using zeros")
            
        if u_y_key:
            u_y_data = mesh.point_data[u_y_key]
            #print(f"  uy data range: [{np.min(u_y_data):.6e}, {np.max(u_y_data):.6e}]")
        else:
            u_y_data = np.zeros(n_points)
            #print(f"  uy data: using zeros")
            
        if temp_key:
            temp_data = mesh.point_data[temp_key]
            #print(f"  T data range: [{np.min(temp_data):.6e}, {np.max(temp_data):.6e}]")
        else:
            temp_data = np.zeros(n_points)
            #print(f"  T data: using zeros")
        
        # Map to grid structure
        for point_idx in range(n_points):
            x_idx, y_idx = coord_to_index[point_idx]
            data_array[x_idx, y_idx, t_idx, 0] = u_x_data[point_idx]  # displacement_x
            data_array[x_idx, y_idx, t_idx, 1] = u_y_data[point_idx]  # displacement_y
            data_array[x_idx, y_idx, t_idx, 2] = temp_data[point_idx] - 273.15  # temperature

    return data_array, coords, time_steps, x_coords, y_coords


def extract_static_displacement_data_einspannung(filename):
    """
    Extract static displacement data from VTU file (no time dependence)
    Returns data organized as (nx, ny, 2) array where 2 components are [u_x, u_y]
    """
    # Load the VTU file
    mesh = pv.read(filename)
    
    # Print mesh summary
    #print(mesh)
    #print("Point data:", mesh.point_data.keys())
    #print("Cell data:", mesh.cell_data.keys())
    
    # Get spatial coordinates
    coords = mesh.points  # shape: (n_points, 3)
    n_points = coords.shape[0]
    
    # Look for displacement field names (without time stamps)
    u_x_key = None
    u_y_key = None
    
    for key in mesh.point_data.keys():
        if "Displacement_field,_X-component" in key and "@_t=" not in key:
            u_x_key = key
        elif "Displacement_field,_Y-component" in key and "@_t=" not in key:
            u_y_key = key
        # Also check for simpler naming conventions
        elif key.lower() in ['u', 'ux', 'displacement_x', 'u_x']:
            u_x_key = key
        elif key.lower() in ['v', 'uy', 'displacement_y', 'u_y']:
            u_y_key = key
    
    #print(f"Found displacement fields - ux: {u_x_key}, uy: {u_y_key}")
    
    if not u_x_key or not u_y_key:
        print("Warning: Could not find both displacement components")
        print("Available fields:", list(mesh.point_data.keys()))
        return None, None, None, None
    
    # Extract displacement data
    u_x_data = mesh.point_data[u_x_key]
    u_y_data = mesh.point_data[u_y_key]
    
    #print(f"ux data range: [{np.min(u_x_data):.6e}, {np.max(u_x_data):.6e}]")
    #print(f"uy data range: [{np.min(u_y_data):.6e}, {np.max(u_y_data):.6e}]")
    
    # Determine grid dimensions (assuming structured grid)
    x_coords = np.unique(coords[:, 0])
    y_coords = np.unique(coords[:, 1])
    nx = len(x_coords)
    ny = len(y_coords)
    
    #print(f"Grid dimensions: nx={nx}, ny={ny}")
    
    # Check if we have a structured grid
    if nx * ny != n_points:
        print(f"Warning: Grid may not be structured. nx*ny={nx*ny}, n_points={n_points}")
        # For unstructured grids, return data as-is with coordinates
        displacement_data = np.column_stack([u_x_data, u_y_data])
        return displacement_data, coords, x_coords, y_coords
    
    # Initialize output array: (nx, ny, 2) for [u_x, u_y]
    data_array = np.zeros((nx, ny, 2))
    
    # Create mapping from coordinates to grid indices
    coord_to_index = {}
    for i, point in enumerate(coords):
        x_idx = np.argmin(np.abs(x_coords - point[0]))
        y_idx = np.argmin(np.abs(y_coords - point[1]))
        coord_to_index[i] = (x_idx, y_idx)
    
    # Map to grid structure
    for point_idx in range(n_points):
        x_idx, y_idx = coord_to_index[point_idx]
        data_array[x_idx, y_idx, 0] = u_x_data[point_idx]  # displacement_x
        data_array[x_idx, y_idx, 1] = u_y_data[point_idx]  # displacement_y
    
    return data_array, coords, x_coords, y_coords


def interpolate_ground_truth(
    ground_truth_tensor: np.ndarray,
    domain: Domain,
) -> np.ndarray:
    """
    Interpolates a low-resolution ground truth tensor onto a high-resolution grid.

    This function is essential for comparing simulation data (e.g., from FEM) with
    model predictions (e.g., from a PINN) when they are defined on different grids.

    Args:
        ground_truth_tensor (np.ndarray): The ground truth data with a shape of
            (nx_low, ny_low, nt_low, n_vars).
        domain (dict): A dictionary specifying the domain boundaries, for example:
            {'x': [x_min, x_max], 'y': [y_min, y_max], 't': [t_min, t_max]}.
        target_resolution (dict): A dictionary specifying the desired output
            resolution, e.g., {'x': nx_high, 'y': ny_high, 't': nt_high}.

    Returns:
        np.ndarray: The interpolated ground truth data on the high-resolution
            grid, with a shape of (nx_high, ny_high, nt_high, n_vars).
    """
    # 1. Define the coordinate vectors for the low-resolution source grid
    nx_low, ny_low, nt_low, n_vars = ground_truth_tensor.shape

    x_low = np.linspace(domain.spatial['x'][0], domain.spatial['x'][1], nx_low)
    y_low = np.linspace(domain.spatial['y'][0], domain.spatial['y'][1], ny_low)
    t_low = np.linspace(domain.temporal['t'][0], domain.temporal['t'][1], nt_low)

    known_points = (x_low, y_low, t_low)

    # 2. Define the high-resolution target grid where we want to find values
    nx_high = domain.resolution['x']
    ny_high = domain.resolution['y']
    nt_high = domain.resolution['t']

    x_high = np.linspace(domain.spatial['x'][0], domain.spatial['x'][1], nx_high)
    y_high = np.linspace(domain.spatial['y'][0], domain.spatial['y'][1], ny_high)
    t_high = np.linspace(domain.temporal['t'][0], domain.temporal['t'][1], nt_high)

    # 3. Create a meshgrid of target points and flatten it for interpn
    X_high, Y_high, T_high = np.meshgrid(x_high, y_high, t_high, indexing='ij')
    target_points = np.stack([X_high.ravel(), Y_high.ravel(), T_high.ravel()], axis=-1)

    # 4. Perform interpolation for each variable (u, v, T)
    interpolated_vars = []
    for i in range(n_vars):
        values = ground_truth_tensor[..., i]
        
        # Perform 3D linear interpolation
        interpolated_flat = interpn(
            points=known_points,
            values=values,
            xi=target_points,
            method='linear',
            bounds_error=False, # Avoid errors for points slightly outside
            fill_value=None      # Use None to enable extrapolation
        )
        interpolated_vars.append(interpolated_flat)

    # 5. Stack the results and reshape to the target grid dimensions
    interpolated_stacked = np.stack(interpolated_vars, axis=-1)
    interpolated_tensor = interpolated_stacked.reshape(nx_high, ny_high, nt_high, n_vars)
    
    return interpolated_tensor