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
