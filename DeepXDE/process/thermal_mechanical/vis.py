import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from process.thermal_mechanical.scale import Scale
from domain_vars import thermal_mechanical_2d_domain
def vis_2d_multi(model, interval=600, **kwargs):
    """
    Creates an animation showing multiple variables in subplots.
    
    Args:
        model: Trained model with predict() method
        variable_indices: List of variables to show [0=u_x, 1=u_y, 2=T]
        interval: Animation delay in ms (default: 600)
    """
    
    # --- Setup ---
    var_names = ['X-Displacement (u)', 'Y-Displacement (v)', 'Temperature (T)']
    cmaps = ['RdBu_r', 'RdBu_r', 'plasma']
    units = ['m', 'm', '°C']
    
    scale = Scale(thermal_mechanical_2d_domain)
    variable_indices = kwargs.get('variable_indices', [0, 1, 2])
    n_vars = len(variable_indices)

    # Domain bounds
    x_start, x_end = thermal_mechanical_2d_domain.spatial['x']
    y_start, y_end = thermal_mechanical_2d_domain.spatial['y']
    t_start, t_end = thermal_mechanical_2d_domain.temporal['t']
    
    # Grid resolution
    nx = kwargs.get('num_x_points', 50)
    ny = kwargs.get('num_y_points', 50)
    nt = kwargs.get('num_t_points', 50)
    
    # --- Generate Data ---
    print(f"Generating {nt} time steps for {n_vars} variables...")
    x_points = np.linspace(x_start, x_end, nx)
    y_points = np.linspace(y_start, y_end, ny)
    t_points = np.linspace(t_start, t_end, nt)
    X, Y = np.meshgrid(x_points, y_points)
    
    # Store data for each variable
    all_data = {idx: [] for idx in variable_indices}

    for i, t in enumerate(t_points):
        if i % 10 == 0:
            print(f"  Step {i+1}/{nt}")
            
        # Create input points
        T_grid = np.full_like(X, t)
        XYT = np.vstack((X.ravel(), Y.ravel(), T_grid.ravel())).T
        
        # Scale and predict
        XYT_scaled = XYT.copy()
        XYT_scaled[:, 0] /= scale.L
        XYT_scaled[:, 1] /= scale.L  
        XYT_scaled[:, 2] /= scale.t
        
        predictions = model.predict(XYT_scaled)
        
        # Process each variable
        for var_idx in variable_indices:
            data = predictions[:, var_idx].reshape(X.shape)
            
            # Unscale for display
            if var_idx == 2:  # Temperature
                data = data * scale.Temperature
                
            all_data[var_idx].append(data)
    
    # Convert to arrays
    for var_idx in variable_indices:
        all_data[var_idx] = np.array(all_data[var_idx])
    
    # --- Create Animation ---
    # Determine subplot layout
    if n_vars == 1:
        rows, cols = 1, 1
        figsize = (8, 6)
    elif n_vars == 2:
        rows, cols = 1, 2
        figsize = (15, 6)
    elif n_vars == 3:
        rows, cols = 1, 3
        figsize = (18, 6)
    else:
        rows, cols = 2, (n_vars + 1) // 2
        figsize = (9 * cols, 6 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_vars == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Initialize plots
    ims = []
    cbars = []
    
    for i, var_idx in enumerate(variable_indices):
        ax = axes[i]
        
        # Color limits
        if var_idx in [0, 1]:  # Displacements - symmetric
            vmax = np.max(np.abs(all_data[var_idx]))
            vmin = -vmax
        else:  # Temperature - full range
            vmin, vmax = all_data[var_idx].min(), all_data[var_idx].max()
        
        # Create plot
        im = ax.imshow(all_data[var_idx][0], 
                      extent=[x_start, x_end, y_start, y_end],
                      origin='lower', aspect='equal', 
                      vmin=vmin, vmax=vmax, cmap=cmaps[var_idx])
        ims.append(im)
        
        # Styling
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(f'{var_names[var_idx]} [{units[var_idx]}]')
        cbars.append(cbar)
        
        ax.set_title(var_names[var_idx])
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)
    
    # Animation function
    def update(frame):
        time_days = t_points[frame] / (24 * 3600)
        fig.suptitle(f'Thermal-Mechanical 2D Animation - Time: {time_days:.2f} days', 
                    fontsize=16, y=0.98)
        
        for i, var_idx in enumerate(variable_indices):
            ims[i].set_array(all_data[var_idx][frame])
        
        return ims
    
    ani = animation.FuncAnimation(fig, update, frames=nt, interval=interval, 
                                  blit=True, repeat=True)
    
    plt.tight_layout()
    return {'field': ani, 'figure': fig}