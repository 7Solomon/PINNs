import torch
import pandas as pd

def load_COMSOL_file_data(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            return file_content
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None
    
def analyze_COMSOL_file_data(file_content):
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
