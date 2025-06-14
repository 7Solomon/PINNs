import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from process.thermal_moisture.scale import Scale
from domain_vars import thermal_moisture_2d_domain

def vis_2d_multi(model, scale: Scale, interval=600, **kwargs):
    """
    Creates an animation showing thermal-moisture 2D results with specialized visualizations.
    
    Args:
        model: Trained model with predict() method
        interval: Animation delay in ms (default: 600)
    """
    
    # Configuration for thermal-moisture specific variables
    var_names = ['Temperature (T)', 'Moisture Content (θ)', 'Temperature Gradient', 'Moisture Gradient', 'Moisture Flow Vectors', 'Phase Diagram']
    cmaps = ['plasma', 'Blues', 'hot', 'viridis', None, None]  # None for vector and scatter plots
    units = ['°C', 'm³/m³', '°C/m', 'm³/m³/m', 'm/s', '']
        
    # Domain and grid setup
    x_start, x_end = thermal_moisture_2d_domain.spatial['x']
    y_start, y_end = thermal_moisture_2d_domain.spatial['y']
    t_start, t_end = thermal_moisture_2d_domain.temporal['t']
    
    nx, ny, nt = 50, 50, 50
    x_points = np.linspace(x_start, x_end, nx)
    y_points = np.linspace(y_start, y_end, ny)
    t_points = np.linspace(t_start, t_end, nt)
    X, Y = np.meshgrid(x_points, y_points)
    
    # Generate predictions for all time steps
    print(f"Generating {nt} time steps...")
    all_predictions = []
    all_gradients = []
    
    for i, t in enumerate(t_points):
        if i % 10 == 0:
            print(f"  Step {i+1}/{nt}")
            
        # Create and scale input points
        T_grid = np.full_like(X, t)
        XYT = np.vstack((X.ravel(), Y.ravel(), T_grid.ravel())).T
        XYT_scaled = XYT / np.array([scale.L, scale.L, scale.t])
        
        # Get predictions and scale back to physical units
        predictions = model.predict(XYT_scaled)
        T_data = predictions[:, 0].reshape(X.shape) * scale.Temperature
        theta_data = predictions[:, 1].reshape(X.shape) * scale.theta
        
        # Calculate gradients for moisture flow visualization
        dx = x_points[1] - x_points[0]
        dy = y_points[1] - y_points[0]
        
        # Temperature gradients
        dT_dx, dT_dy = np.gradient(T_data, dx, dy)
        T_grad_mag = np.sqrt(dT_dx**2 + dT_dy**2)
        
        # Moisture gradients
        dtheta_dx, dtheta_dy = np.gradient(theta_data, dx, dy)
        theta_grad_mag = np.sqrt(dtheta_dx**2 + dtheta_dy**2)
        
        all_predictions.append([T_data, theta_data, T_grad_mag, theta_grad_mag])
        all_gradients.append([dT_dx, dT_dy, dtheta_dx, dtheta_dy])
    
    all_predictions = np.array(all_predictions)  # Shape: (nt, 4, ny, nx)
    all_gradients = np.array(all_gradients)     # Shape: (nt, 4, ny, nx)
    
    # Setup figure with 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Initialize field plots (first 4 subplots)
    field_ims = []
    for i in range(4):
        ax = axes[i]
        
        # Set color limits
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
    
    # Setup moisture flow vectors plot (subplot 5)
    vector_ax = axes[4]
    # Subsample for cleaner vector field
    vector_subsample = 5
    X_vec = X[::vector_subsample, ::vector_subsample]
    Y_vec = Y[::vector_subsample, ::vector_subsample]
    
    # Initialize quiver plot
    quiver = vector_ax.quiver(X_vec, Y_vec, 
                             np.zeros_like(X_vec), np.zeros_like(Y_vec),
                             scale=1e-6, alpha=0.7, color='blue')
    vector_ax.set_title('Moisture Flow Vectors')
    vector_ax.set_xlabel('X Position [m]')
    vector_ax.set_ylabel('Y Position [m]')
    vector_ax.set_aspect('equal')
    vector_ax.grid(True, alpha=0.3)
    
    # Setup T-θ phase diagram (subplot 6)
    phase_ax = axes[5]
    # Sample points for phase diagram
    sample_indices = np.random.choice(nx*ny, size=200, replace=False)
    scatter = phase_ax.scatter([], [], c=[], cmap='viridis', alpha=0.6, s=20)
    phase_ax.set_xlabel('Temperature [°C]')
    phase_ax.set_ylabel('Moisture Content [m³/m³]')
    phase_ax.set_title('T-θ Phase Evolution')
    phase_ax.grid(True, alpha=0.3)
    
    # Set phase diagram limits
    T_all = all_predictions[:, 0].flatten()
    theta_all = all_predictions[:, 1].flatten()
    phase_ax.set_xlim(T_all.min(), T_all.max())
    phase_ax.set_ylim(theta_all.min(), theta_all.max())
    
    def update_frame(frame):
        # Update title with current time
        time_days = t_points[frame] / (24 * 3600)
        fig.suptitle(f'Thermal-Moisture 2D Animation - Time: {time_days:.2f} days', 
                    fontsize=16, y=0.95)
        
        # Update field plots
        for i in range(4):
            field_ims[i].set_array(all_predictions[frame, i])
        
        # Update moisture flow vectors
        dtheta_dx_vec = all_gradients[frame, 2][::vector_subsample, ::vector_subsample]
        dtheta_dy_vec = all_gradients[frame, 3][::vector_subsample, ::vector_subsample]
        
        # Scale vectors for visibility (moisture flows are typically small)
        scale_factor = 1e6
        quiver.set_UVC(dtheta_dx_vec * scale_factor, dtheta_dy_vec * scale_factor)
        
        # Update phase diagram
        T_sample = all_predictions[frame, 0].flatten()[sample_indices]
        theta_sample = all_predictions[frame, 1].flatten()[sample_indices]
        time_colors = np.full_like(T_sample, frame)  # Color by time
        
        # Update scatter plot data
        scatter.set_offsets(np.column_stack([T_sample, theta_sample]))
        scatter.set_array(time_colors)
        
        return field_ims + [quiver, scatter]
    
    # Create animation
    ani = animation.FuncAnimation(fig, update_frame, frames=nt, interval=interval, 
                                  blit=False, repeat=True)
    
    plt.tight_layout()
    return {'field': ani, 'figure': fig}

def vis_moisture_profiles(model, times=[0, 0.25, 0.5, 0.75, 1.0], **kwargs):
    """
    Creates moisture and temperature profiles at different times along a centerline.
    
    Args:
        model: Trained model
        times: Relative times (0-1) to show profiles
    """
    scale = Scale(thermal_moisture_2d_domain)
    
    x_start, x_end = thermal_moisture_2d_domain.spatial['x']
    y_start, y_end = thermal_moisture_2d_domain.spatial['y']
    t_start, t_end = thermal_moisture_2d_domain.temporal['t']
    
    # Create centerline through domain
    x_line = np.linspace(x_start, x_end, 100)
    y_center = (y_start + y_end) / 2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
    
    for i, rel_time in enumerate(times):
        t_actual = t_start + rel_time * (t_end - t_start)
        
        # Create input points
        XYT = np.column_stack([x_line, 
                              np.full_like(x_line, y_center), 
                              np.full_like(x_line, t_actual)])
        XYT_scaled = XYT / np.array([scale.L, scale.L, scale.t])
        
        # Get predictions
        predictions = model.predict(XYT_scaled)
        T_line = predictions[:, 0] * scale.Temperature
        theta_line = predictions[:, 1] * scale.theta
        
        # Plot profiles
        time_days = t_actual / (24 * 3600)
        ax1.plot(x_line, T_line, color=colors[i], 
                label=f't = {time_days:.1f} days', linewidth=2)
        ax2.plot(x_line, theta_line, color=colors[i], 
                label=f't = {time_days:.1f} days', linewidth=2)
    
    ax1.set_xlabel('X Position [m]')
    ax1.set_ylabel('Temperature [°C]')
    ax1.set_title('Temperature Profiles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('X Position [m]')
    ax2.set_ylabel('Moisture Content [m³/m³]')
    ax2.set_title('Moisture Content Profiles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig