import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# Removed ipywidgets and IPython.display imports if they were still there

def draw_time_dependant_plot(data, fps=15, cmap='viridis'):
    X = data['X']
    Y = data['Y']
    T = data['T']
    output = data['output']

    if output.ndim != 3:
        print(f"Error: Expected 'output' data to be 3D (nx, ny, nt), but got shape {output.shape}")
        return

    num_x, num_y, num_t = output.shape
    print(f"Data dimensions: nx={num_x}, ny={num_y}, nt={num_t}")
    print(f"Generating animation with {num_t} frames...")

    # --- Create Plot Elements ---
    fig, ax = plt.subplots(figsize=(5, 8))

    # Initial plot with the first time step (t=0)
    x_coords = X[:, :, 0]
    y_coords = Y[:, :, 0]
    z_data = output[:, :, 0]
    time_val = T[0, 0, 0] # Get time value for the first slice

    # Limits for color
    vmin = np.min(output)
    vmax = np.max(output)

    quadmesh = ax.pcolormesh(x_coords, y_coords, z_data, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(quadmesh, ax=ax)
    cbar.set_label('')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    time_text = ax.set_title(f'Time: {time_val:.3f}')

    def update_plot(frame_index):
        z_data = output[:, :, frame_index]
        time_val = T[0, 0, frame_index]
        quadmesh.set_array(z_data.ravel())
        time_text.set_text(f'Time: {time_val:.3f}')
        return [quadmesh, time_text]

    # --- Create Animation ---
    # Note: blit=True can be faster but sometimes causes issues with display.
    # If the animation window is blank or behaves strangely, try blit=False.
    ani = animation.FuncAnimation(fig, update_plot, frames=num_t,
                                  interval=1000/fps, repeat=True)

    plt.show() 


def draw_zt_time_dependant_plot(data, fps=15, ylabels=None, title_prefix='Profile'):
    """
    Animates multiple 1D profiles along the Z-axis over time T,
    each in its own subplot.

    Args:
        data (dict): Dictionary containing 'Z', 'T', and 'output' numpy arrays.
                     'Z' should be (nz, nt), 'T' should be (nz, nt),
                     and 'output' should be a *list* of 2D arrays,
                     each with shape (nz, nt).
        fps (int): Frames per second for the animation.
        ylabels (list[str], optional): List of labels for the y-axis of each subplot.
                                      If None, a generic label is used.
        title_prefix (str): Prefix for the plot title (e.g., 'Pressure Head Profile').
    """
    Z = data['Z']
    T = data['T']
    output_list = data['output'] # Expecting a list now

    if not isinstance(output_list, list):
        print(f"Error: Expected 'output' to be a list of 2D arrays.")
        return
    if not output_list:
        print("Error: 'output' list is empty.")
        return

    num_plots = len(output_list)
    if ylabels is None:
        ylabels = [f'Output {i+1}' for i in range(num_plots)]
    elif len(ylabels) != num_plots:
        print("Warning: Number of ylabels does not match number of outputs. Using generic labels.")
        ylabels = [f'Output {i+1}' for i in range(num_plots)]


    # --- Validate Shapes ---
    first_output_shape = None
    for i, output in enumerate(output_list):
        if output.ndim != 2:
            print(f"Error: Expected output element {i} to be 2D (nz, nt), but got shape {output.shape}")
            return
        if i == 0:
            first_output_shape = output.shape
            if Z.shape != first_output_shape or T.shape != first_output_shape:
                 print(f"Error: Shape mismatch for first output. Z: {Z.shape}, T: {T.shape}, output[0]: {first_output_shape}")
                 return
        elif output.shape != first_output_shape:
            print(f"Error: Shape mismatch between output elements. output[0]: {first_output_shape}, output[{i}]: {output.shape}")
            return

    num_z, num_t = first_output_shape
    print(f"Data dimensions: nz={num_z}, nt={num_t}")
    print(f"Number of plots: {num_plots}")
    print(f"Generating animation with {num_t} frames...")

    # --- Create Plot Elements ---
    # Create subplots vertically, sharing the x-axis (Z)
    fig, axes = plt.subplots(num_plots, 1, figsize=(6, 4 * num_plots), sharex=True)
    # Ensure axes is always an array, even if num_plots is 1
    axes = np.atleast_1d(axes)

    # Z coordinates (constant for each time step)
    z_coords = Z[:, 0]
    # Time values (use the first row, assuming T varies along columns)
    time_values = T[0, :]

    lines = []
    time_texts = []

    for i, ax in enumerate(axes):
        output = output_list[i]
        ylabel = ylabels[i]

        # Initial plot with the first time step (t=0)
        output_at_t0 = output[:, 0]
        time_val = time_values[0]

        # Determine plot limits based on the entire dataset for this specific output
        min_val = np.min(output)
        max_val = np.max(output)
        # Add some padding, handle cases where min/max are zero or same
        padding = 0.1 * max(abs(min_val), abs(max_val), 1e-6) # Avoid division by zero if range is tiny
        ax.set_ylim(min_val - padding, max_val + padding)

        line, = ax.plot(z_coords, output_at_t0) # Store the line object
        lines.append(line)

        ax.set_ylabel(ylabel)
        title_obj = ax.set_title(f'{title_prefix} {i+1} at Time: {time_val:.3f}')
        time_texts.append(title_obj)
        ax.grid(True)

    axes[-1].set_xlabel('Z Coordinate')

    def update_plot(frame_index):
        """Updates the plot for a given time frame."""
        artists = []
        time_val = time_values[frame_index]
        for i, line in enumerate(lines):
            output = output_list[i]
            output_at_t = output[:, frame_index]
            line.set_ydata(output_at_t) # Update the y-data of the existing line
            time_texts[i].set_text(f'{title_prefix} {i+1} at Time: {time_val:.3f}')
            artists.append(line)
            artists.append(time_texts[i])
        return artists # Return list of all updated artists

    # --- Create Animation ---
    ani = animation.FuncAnimation(fig, update_plot, frames=num_t,
                                  interval=1000/fps, blit=True, repeat=True)

    plt.tight_layout()
    plt.show()

