import numpy as np
import deepxde as dde

def load_loss_history_object(fname):
    """
    Loads loss history from a text file saved by dde.utils.external.save_loss_history
    and reconstructs a deepxde.callbacks.LossHistory object.
    """
    try:
        data = np.loadtxt(fname, skiprows=1)
    except Exception as e:
        print(f"Error loading loss history file {fname}: {e}")
        return None

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] == 0:
        print(f"Warning: Loss history file {fname} is empty or contains no data rows.")
        lh = dde.model.LossHistory()
        lh.steps = []
        lh.loss_train = []
        lh.loss_test = []
        lh.metrics_test = []
        return lh

    steps = data[:, 0].tolist()
    
    # Get the number of columns for each component
    # For this example, we'll use the header to determine column counts
    try:
        with open(fname, 'r') as f:
            header_line = f.readline().strip() # Read and strip the whole line
            if header_line.startswith('#'):
                header_line = header_line[1:].strip() # Remove '#' and then strip again
            header = [h.strip() for h in header_line.split(',')] # Split by comma and strip each item
        # Count occurrences of each component type
        loss_train_count = header.count('loss_train')
        loss_test_count = header.count('loss_test')
        metrics_test_count = header.count('metrics_test')
        
        # If header doesn't contain proper counts, use these defaults
        if loss_train_count == 0: loss_train_count = 5
        if loss_test_count == 0: loss_test_count = 5
    except:
        # Default values if header parsing fails
        loss_train_count = 5
        loss_test_count = 5
    
    # Extract train loss components
    current_col_idx = 1
    if data.shape[1] > current_col_idx + loss_train_count - 1:
        end_train_idx = current_col_idx + loss_train_count
        loss_train_values = data[:, current_col_idx:end_train_idx].tolist()
        current_col_idx = end_train_idx
    else:
        print(f"Warning: Not enough columns for all train loss components in {fname}.")
        loss_train_values = [[0.0] * loss_train_count for _ in range(len(steps))]

    # Extract test loss components
    if data.shape[1] > current_col_idx + loss_test_count - 1:
        end_test_idx = current_col_idx + loss_test_count
        loss_test_values = data[:, current_col_idx:end_test_idx].tolist()
        current_col_idx = end_test_idx
    else:
        print(f"Warning: Not enough columns for all test loss components in {fname}.")
        loss_test_values = [[0.0] * loss_test_count for _ in range(len(steps))]
        
    # Remaining columns are metrics_test
    if data.shape[1] > current_col_idx:
        metrics_test_values = data[:, current_col_idx:].tolist()
    else:
        metrics_test_values = [[] for _ in range(len(steps))]

    lh = dde.model.LossHistory()
    lh.steps = steps
    lh.loss_train = loss_train_values
    lh.loss_test = loss_test_values
    lh.metrics_test = metrics_test_values

    return lh