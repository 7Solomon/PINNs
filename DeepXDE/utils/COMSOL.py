import torch
import pandas as pd
import os
import numpy as np

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

#def analyze_transient_COMSOL_file_data(file_content):
#    # split into raw lines
#    raw_lines = file_content.splitlines()
#
#    # strip leading ‘% ’ and any extra whitespace
#    lines = [l.lstrip('% ').strip() for l in raw_lines]
#
#    # 1) locate the header now that any ‘% ’ is gone
#    header_idx = next(
#        (i for i, l in enumerate(lines)
#         if l.startswith('X') and 'T (K)' in l and '@' in l),
#        None
#    )
#    if header_idx is None:
#        raise ValueError("Could not find transient data header in COMSOL file.")
#
#    # 2) pull out the header tokens
#    parts = lines[header_idx].split()
#    time_tokens = parts[5::4]
#    times = [float(tok.split('=')[1]) for tok in time_tokens]
#
#    # 3) parse the following data lines
#    coords = []
#    temps = []
#    for row in lines[header_idx + 1:]:
#        vals = row.split()
#        if len(vals) != 2 + len(times):
#            continue
#        nums = [float(v) for v in vals]
#        coords.append(nums[:2])
#        temps.append(nums[2:])
#
#    # 4) build tensors and convert K→°C
#    domain_tensor = torch.tensor(coords, dtype=torch.float32, device='cpu')
#    time_tensor = torch.tensor(times, dtype=torch.float32, device='cpu')
#    temp_tensor = torch.tensor(temps, dtype=torch.float32, device='cpu') - 273.15
#
#    return domain_tensor, time_tensor, temp_tensor