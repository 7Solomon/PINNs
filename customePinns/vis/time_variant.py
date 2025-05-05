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


def draw_zt_time_dependant_plot(data, fps=15, ylabel='Output Value', title_prefix='Profile'):
    """
    Animates a 1D profile along the Z-axis over time T.

    Args:
        data (dict): Dictionary containing 'Z', 'T', and 'output' numpy arrays.
                     'Z' should be (nz, nt), 'T' should be (nz, nt),
                     and 'output' should be (nz, nt).
        fps (int): Frames per second for the animation.
        ylabel (str): Label for the y-axis of the plot.
        title_prefix (str): Prefix for the plot title (e.g., 'Pressure Head Profile').
    """
    Z = data['Z']
    T = data['T']
    output = data['output']

    if output.ndim != 2:
        print(f"Error: Expected 'output' data to be 2D (nz, nt), but got shape {output.shape}")
        return
    if Z.shape != output.shape or T.shape != output.shape:
         print(f"Error: Shape mismatch. Z: {Z.shape}, T: {T.shape}, output: {output.shape}")
         return

    num_z, num_t = output.shape
    print(f"Data dimensions: nz={num_z}, nt={num_t}")
    print(f"Generating animation with {num_t} frames...")

    # --- Create Plot Elements ---
    fig, ax = plt.subplots(figsize=(6, 5))

    # Z coordinates (constant for each time step)
    z_coords = Z[:, 0]
    # Time values (use the first row, assuming T varies along columns)
    time_values = T[0, :]

    # Initial plot with the first time step (t=0)
    output_at_t0 = output[:, 0]
    time_val = time_values[0]

    # Determine plot limits based on the entire dataset
    min_val = np.min(output)
    max_val = np.max(output)
    ax.set_ylim(min_val - 0.1 * abs(min_val), max_val + 0.1 * abs(max_val)) # Add some padding

    line, = ax.plot(z_coords, output_at_t0) # Store the line object

    ax.set_xlabel('Z Coordinate')
    ax.set_ylabel(ylabel)
    time_text = ax.set_title(f'{title_prefix} at Time: {time_val:.3f}')
    ax.grid(True)

    def update_plot(frame_index):
        """Updates the plot for a given time frame."""
        output_at_t = output[:, frame_index]
        time_val = time_values[frame_index]
        line.set_ydata(output_at_t) # Update the y-data of the existing line
        time_text.set_text(f'{title_prefix} at Time: {time_val:.3f}')
        return [line, time_text]

    # --- Create Animation ---
    ani = animation.FuncAnimation(fig, update_plot, frames=num_t,
                                  interval=1000/fps, blit=True, repeat=True)

    plt.tight_layout() # Adjust layout
    plt.show()

