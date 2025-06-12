import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from process.thermal_mechanical.scale import Scale

from config import concreteData
from domain_vars import thermal_mechanical_2d_domain
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from process.thermal_mechanical.scale import Scale
from config import concreteData
from domain_vars import thermal_mechanical_2d_domain

def vis_2d_multi(model, interval=600, **kwargs):
    """
    Creates an animation showing thermal-mechanical 2D results with all variables.
    
    Args:
        model: Trained model with predict() method
        interval: Animation delay in ms (default: 600)
    """
    
    # Configuration
    var_names = ['X-Displacement (u)', 'Y-Displacement (v)', 'Temperature (T)', 'Displacement Magnitude']
    cmaps = ['RdBu_r', 'RdBu_r', 'plasma', 'viridis']
    units = ['m', 'm', '°C', 'm']
    
    scale = Scale(thermal_mechanical_2d_domain)
    
    # Domain and grid setup
    x_start, x_end = thermal_mechanical_2d_domain.spatial['x']
    y_start, y_end = thermal_mechanical_2d_domain.spatial['y']
    t_start, t_end = thermal_mechanical_2d_domain.temporal['t']
    
    nx, ny, nt = 50, 50, 50
    x_points = np.linspace(x_start, x_end, nx)
    y_points = np.linspace(y_start, y_end, ny)
    t_points = np.linspace(t_start, t_end, nt)
    X, Y = np.meshgrid(x_points, y_points)
    
    # Generate predictions for all time steps
    print(f"Generating {nt} time steps...")
    all_predictions = []
    
    for i, t in enumerate(t_points):
        if i % 10 == 0:
            print(f"  Step {i+1}/{nt}")
            
        # Create and scale input points
        T_grid = np.full_like(X, t)
        XYT = np.vstack((X.ravel(), Y.ravel(), T_grid.ravel())).T
        XYT_scaled = XYT / np.array([scale.L, scale.L, scale.t])
        
        # Get predictions and scale back to physical units
        predictions = model.predict(XYT_scaled)
        u_data = predictions[:, 0].reshape(X.shape) * scale.U
        v_data = predictions[:, 1].reshape(X.shape) * scale.U
        T_data = predictions[:, 2].reshape(X.shape) * scale.Temperature
        mag_data = np.sqrt(u_data**2 + v_data**2)
        
        all_predictions.append([u_data, v_data, T_data, mag_data])
    
    all_predictions = np.array(all_predictions)  # Shape: (nt, 4, ny, nx)
    
    # Setup figure with 2x3 layout (4 field plots + 2 mesh plots)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Initialize field plots (first 4 subplots)
    field_ims = []
    for i in range(4):
        ax = axes[i]
        
        # Set color limits
        if i in [0, 1]:  # Displacements - symmetric around zero
            vmax = np.max(np.abs(all_predictions[:, i]))
            vmin = -vmax
        else:  # Temperature and magnitude - full range
            vmin = all_predictions[:, i].min()
            vmax = all_predictions[:, i].max()
        
        # Create initial plot
        im = ax.imshow(all_predictions[0, i], 
                      extent=[x_start, x_end, y_start, y_end],
                      origin='lower', aspect='equal', 
                      vmin=vmin, vmax=vmax, cmap=cmaps[i])
        field_ims.append(im)
        
        # Add colorbar and styling
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(f'{var_names[i]} [{units[i]}]')
        ax.set_title(var_names[i])
        ax.set_xlabel('X Position [m]')
        ax.set_ylabel('Y Position [m]')
        ax.grid(True, alpha=0.3)
    
    # Setup mesh deformation plots (subplots 5 and 6)
    mesh_subsample = 8  # Every 8th point for cleaner mesh
    X_mesh = X[::mesh_subsample, ::mesh_subsample] 
    Y_mesh = Y[::mesh_subsample, ::mesh_subsample]
    
    # Original mesh plot (subplot 5)
    original_ax = axes[4]
    original_ax.plot(X_mesh, Y_mesh, 'k-', alpha=0.5, linewidth=0.8)
    original_ax.plot(X_mesh.T, Y_mesh.T, 'k-', alpha=0.5, linewidth=0.8)
    original_ax.set_title('Original Mesh')
    original_ax.set_xlabel('X Position [m]')
    original_ax.set_ylabel('Y Position [m]')
    original_ax.set_aspect('equal')
    original_ax.grid(True, alpha=0.3)
    
    # Deformed mesh plot (subplot 6)
    deformed_ax = axes[5]
    deformed_lines_h = []  # Horizontal lines
    deformed_lines_v = []  # Vertical lines
    
    # Initialize deformed mesh lines
    amplification = 1000  # Amplify displacements for visibility
    for i in range(X_mesh.shape[0]):
        line, = deformed_ax.plot([], [], 'r-', linewidth=1)
        deformed_lines_h.append(line)
    for j in range(X_mesh.shape[1]):
        line, = deformed_ax.plot([], [], 'r-', linewidth=1)
        deformed_lines_v.append(line)
    
    deformed_ax.set_title(f'Deformed Mesh ({amplification}x amplified)')
    deformed_ax.set_xlabel('X Position [m]')
    deformed_ax.set_ylabel('Y Position [m]')
    deformed_ax.set_aspect('equal')
    deformed_ax.grid(True, alpha=0.3)
    
    def update_frame(frame):
        # Update title with current time
        time_days = t_points[frame] / (24 * 3600)
        fig.suptitle(f'Thermal-Mechanical 2D Animation - Time: {time_days:.2f} days', 
                    fontsize=16, y=0.95)
        
        # Update field plots
        for i in range(4):
            field_ims[i].set_array(all_predictions[frame, i])
        
        # Update deformed mesh
        u_mesh = all_predictions[frame, 0][::mesh_subsample, ::mesh_subsample] * amplification
        v_mesh = all_predictions[frame, 1][::mesh_subsample, ::mesh_subsample] * amplification
        X_deformed = X_mesh + u_mesh
        Y_deformed = Y_mesh + v_mesh
        
        # Update horizontal lines
        for i, line in enumerate(deformed_lines_h):
            line.set_data(X_deformed[i, :], Y_deformed[i, :])
        
        # Update vertical lines  
        for j, line in enumerate(deformed_lines_v):
            line.set_data(X_deformed[:, j], Y_deformed[:, j])
        
        # Adjust deformed mesh axis limits
        x_range = X_deformed.max() - X_deformed.min()
        y_range = Y_deformed.max() - Y_deformed.min()
        margin = 0.1
        deformed_ax.set_xlim(X_deformed.min() - margin * x_range, 
                           X_deformed.max() + margin * x_range)
        deformed_ax.set_ylim(Y_deformed.min() - margin * y_range, 
                           Y_deformed.max() + margin * y_range)
        
        return field_ims + deformed_lines_h + deformed_lines_v
    
    # Create animation
    ani = animation.FuncAnimation(fig, update_frame, frames=nt, interval=interval, 
                                  blit=False, repeat=True)
    
    plt.tight_layout()
    return {'field': ani, 'figure': fig}