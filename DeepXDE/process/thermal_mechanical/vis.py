import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from process.thermal_mechanical.scale import Scale

from domain_vars import thermal_mechanical_2d_domain
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from process.thermal_mechanical.scale import Scale
from domain_vars import thermal_mechanical_2d_domain

def vis_2d_multi(model, scale: Scale, interval=600, **kwargs):
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
    
    ##### Calculate amplification factor for visualization
    max_u_abs = np.max(np.abs(all_predictions[:, 0, :, :]))
    max_v_abs = np.max(np.abs(all_predictions[:, 1, :, :]))
    
    all_u_data = all_predictions[:, 0, :, :]
    all_v_data = all_predictions[:, 1, :, :]
    all_displacement_magnitudes = np.sqrt(all_u_data**2 + all_v_data**2)
    max_actual_displacement = np.max(all_displacement_magnitudes)
    
    target_visual_displacement = 0.05  # Target for the maximum displayed deformation (in meters)
    
    if max_actual_displacement > 1e-9:  # Avoid division by zero or very small numbers
        amplification = target_visual_displacement / max_actual_displacement
    else:
        amplification = 10  # Default amplification if displacements are negligible
        
    # Optional: Cap the amplification factor to a reasonable range
    min_amplification = 1.0
    max_amplification = 500.0 # Adjust as needed
    amplification = int(np.clip(amplification, min_amplification, max_amplification))
    


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
    
    # Combined Original and Deformed Mesh Plot (subplot 5, using axes[4])
    combined_mesh_ax = axes[4] # Use the 5th subplot (index 4)
    
    # Plot original mesh as a reference (lighter color, thinner lines)
    combined_mesh_ax.plot(X_mesh, Y_mesh, 'k-', alpha=0.3, linewidth=0.7, label='Original Mesh')
    combined_mesh_ax.plot(X_mesh.T, Y_mesh.T, 'k-', alpha=0.3, linewidth=0.7)
    
    deformed_lines_h = []  # Horizontal lines for deformed mesh
    deformed_lines_v = []  # Vertical lines for deformed mesh
    
    # Initialize deformed mesh lines (more prominent color)
    for i in range(X_mesh.shape[0]):
        line, = combined_mesh_ax.plot([], [], 'r-', linewidth=1.2, label='Deformed Mesh' if i == 0 else "") # Label only once
        deformed_lines_h.append(line)
    for j in range(X_mesh.shape[1]):
        line, = combined_mesh_ax.plot([], [], 'r-', linewidth=1.2)
        deformed_lines_v.append(line)
    
    combined_mesh_ax.set_title(f'Original & Deformed Mesh ({amplification}x amplified)')
    combined_mesh_ax.set_xlabel('X Position [m]')
    combined_mesh_ax.set_ylabel('Y Position [m]')
    combined_mesh_ax.set_aspect('equal')
    combined_mesh_ax.grid(True, alpha=0.3)

    # make last subplot invisible
    fig.delaxes(axes[5])
    
    def update_frame(frame):
        # Update title with current time
        time_days = t_points[frame] / (24 * 3600)
        fig.suptitle(f'Thermal-Mechanical 2D Animation - Time: {time_days:.2f} days', 
                    fontsize=16, y=0.95) # Adjusted y for suptitle if layout changes
        
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
        
        # Adjust combined mesh axis limits to encompass both original and deformed states
        # Consider the extent of both X_mesh, Y_mesh and X_deformed, Y_deformed
        all_x_points = np.concatenate([X_mesh.ravel(), X_deformed.ravel()])
        all_y_points = np.concatenate([Y_mesh.ravel(), Y_deformed.ravel()])

        x_min_plot, x_max_plot = all_x_points.min(), all_x_points.max()
        y_min_plot, y_max_plot = all_y_points.min(), all_y_points.max()
        
        x_range = x_max_plot - x_min_plot
        y_range = y_max_plot - y_min_plot
        margin = 0.1 # Keep a small margin

        combined_mesh_ax.set_xlim(x_min_plot - margin * x_range, 
                                  x_max_plot + margin * x_range)
        combined_mesh_ax.set_ylim(y_min_plot - margin * y_range, 
                                  y_max_plot + margin * y_range)
        
        return field_ims + deformed_lines_h + deformed_lines_v
    
    # Create animation
    ani = animation.FuncAnimation(fig, update_frame, frames=nt, interval=interval, 
                                  blit=False, repeat=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit suptitle
    return {'field': ani, 'figure': fig}