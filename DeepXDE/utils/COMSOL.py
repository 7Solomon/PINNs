import torch
import pandas as pd
import os



def load_data_from_file(file_path, delimiter=',', skiprows=0):
    """
    Loads numerical data from a text file into a NumPy array.

    Args:
        file_path (str): The path to the data file.
        delimiter (str, optional): The string used to separate values. Defaults to ','.
        skiprows (int, optional): Skip the first `skiprows` lines, e.g., for headers. Defaults to 0.

    Returns:
        np.ndarray: A NumPy array containing the loaded data.
                    Returns None if the file is not found or an error occurs during loading.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    try:
        data = np.loadtxt(file_path, delimiter=delimiter, skiprows=skiprows)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


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


def analyze_transient_COMSOL_file_data(file_content):
    # split into raw lines
    raw_lines = file_content.splitlines()

    # strip leading ‘% ’ and any extra whitespace
    lines = [l.lstrip('% ').strip() for l in raw_lines]

    # 1) locate the header now that any ‘% ’ is gone
    header_idx = next(
        (i for i, l in enumerate(lines)
         if l.startswith('X') and 'T (K)' in l and '@' in l),
        None
    )
    if header_idx is None:
        raise ValueError("Could not find transient data header in COMSOL file.")

    # 2) pull out the header tokens
    parts = lines[header_idx].split()
    time_tokens = parts[5::4]
    times = [float(tok.split('=')[1]) for tok in time_tokens]

    # 3) parse the following data lines
    coords = []
    temps = []
    for row in lines[header_idx + 1:]:
        vals = row.split()
        if len(vals) != 2 + len(times):
            continue
        nums = [float(v) for v in vals]
        coords.append(nums[:2])
        temps.append(nums[2:])

    # 4) build tensors and convert K→°C
    domain_tensor = torch.tensor(coords, dtype=torch.float32, device='cpu')
    time_tensor = torch.tensor(times, dtype=torch.float32, device='cpu')
    temp_tensor = torch.tensor(temps, dtype=torch.float32, device='cpu') - 273.15

    return domain_tensor, time_tensor, temp_tensor