import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from process.mechanical_moisture.scale import Scale
from utils.metadata import Domain # Assuming Domain is in utils.metadata

from domain_vars import mechanical_moisture_2d_domain

def vis_2d_mechanical_moisture(model, scale: Scale, interval=600, **kwargs):
    """
    Creates an animation showing mechanical-moisture 2D results.
    
    Args:
        model: Trained model with predict() method.
        scale: Scale object for unscaling variables.
        interval: Animation delay in ms (default: 600).
    """
    
    var_names = ['u-displacement', 'v-displacement', 'Moisture Content (θ)', 
                 'Volumetric Strain (ε_v)', 'Shear Strain (γ_xy)', 'Displacement Vectors']
    cmaps = ['viridis', 'viridis', 'Blues', 'coolwarm', 'coolwarm', None] 
    units = ['m', 'm', 'm³/m³', '-', '-', 'm']
        
    x_start, x_end = mechanical_moisture_2d_domain.spatial['x']
    y_start, y_end = mechanical_moisture_2d_domain.spatial['y']
    t_start, t_end = mechanical_moisture_2d_domain.temporal['t']
    
    nx, ny, nt = 50, 50, 30 # Reduced nt for faster generation, adjust as needed
    x_points = np.linspace(x_start, x_end, nx)
    y_points = np.linspace(y_start, y_end, ny)
    t_points = np.linspace(t_start, t_end, nt)
    X, Y = np.meshgrid(x_points, y_points)
    
    print(f"Generating {nt} time steps for mechanical-moisture visualization...")
    all_plot_data = [] # To store [u, v, theta, vol_strain, g_xy] for each time step
    
    for i, t in enumerate(t_points):
        if i % 5 == 0 or i == nt -1 :
            print(f"  Step {i+1}/{nt}")
            
        T_grid = np.full_like(X, t)
        XYT = np.vstack((X.ravel(), Y.ravel(), T_grid.ravel())).T
        XYT_scaled = XYT / np.array([scale.L, scale.L, scale.t])
        
        predictions = model.predict(XYT_scaled) # Should output u_scaled, v_scaled, theta_scaled
        
        u_data = predictions[:, 0].reshape(X.shape) * scale.epsilon
        v_data = predictions[:, 1].reshape(X.shape) * scale.epsilon
        theta_data = predictions[:, 2].reshape(X.shape) * scale.theta
        
        dx = x_points[1] - x_points[0]
        dy = y_points[1] - y_points[0]
        
        du_dx, du_dy = np.gradient(u_data, dx, dy, axis=(1,0)) # d/dx is change along columns, d/dy along rows
        dv_dx, dv_dy = np.gradient(v_data, dx, dy, axis=(1,0))
        
        e_x = du_dx
        e_y = dv_dy
        g_xy = du_dy + dv_dx # Engineering shear strain
        
        vol_strain = e_x + e_y
        
        all_plot_data.append([u_data, v_data, theta_data, vol_strain, g_xy])
    
    all_plot_data = np.array(all_plot_data)  # Shape: (nt, 5, ny, nx)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    field_ims = []
    for i in range(5): # For u, v, theta, vol_strain, g_xy
        ax = axes[i]
        
        vmin = all_plot_data[:, i].min()
        vmax = all_plot_data[:, i].max()
        # Ensure vmin and vmax are different for stable color mapping
        if vmin == vmax:
            vmin -= 1e-9 # Add small epsilon
            vmax += 1e-9
            
        im = ax.imshow(all_plot_data[0, i], 
                      extent=[x_start, x_end, y_start, y_end],
                      origin='lower', aspect='equal',
                      vmin=vmin, vmax=vmax, cmap=cmaps[i])
        field_ims.append(im)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(f'{var_names[i]} [{units[i]}]')
        ax.set_title(var_names[i])
        ax.set_xlabel('X Position [m]')
        ax.set_ylabel('Y Position [m]')
        ax.grid(True, alpha=0.3)
    
    # Displacement vectors plot (subplot 6)
    vector_ax = axes[5]
    vector_subsample = 5 
    X_vec = X[::vector_subsample, ::vector_subsample]
    Y_vec = Y[::vector_subsample, ::vector_subsample]
    
    # Initial quiver with placeholder zeros, scale will be dynamic or fixed
    # Determine a reasonable scale for quiver based on max displacement magnitude
    max_disp_magnitude = np.sqrt(all_plot_data[0,0]**2 + all_plot_data[0,1]**2).max()
    quiver_scale_val = max_disp_magnitude * 20 # Heuristic, adjust as needed
    if quiver_scale_val == 0: quiver_scale_val = 1.0


    quiver = vector_ax.quiver(X_vec, Y_vec, 
                             all_plot_data[0, 0][::vector_subsample, ::vector_subsample], # u-component
                             all_plot_data[0, 1][::vector_subsample, ::vector_subsample], # v-component
                             scale=quiver_scale_val, angles='xy', scale_units='xy', color='blue', alpha=0.7)
    vector_ax.set_title(var_names[5])
    vector_ax.set_xlabel('X Position [m]')
    vector_ax.set_ylabel('Y Position [m]')
    vector_ax.set_aspect('equal') # Keep aspect equal for vectors
    vector_ax.set_xlim(x_start, x_end)
    vector_ax.set_ylim(y_start, y_end)
    vector_ax.grid(True, alpha=0.3)
    
    def update_frame(frame):
        time_val = t_points[frame] # Assuming t_points are in seconds
        fig.suptitle(f'Mechanical-Moisture 2D Animation - Time: {time_val:.2e} s', 
                    fontsize=16, y=0.98) # Adjusted y for suptitle
        
        for i in range(5):
            field_ims[i].set_array(all_plot_data[frame, i])
        
        u_vec_data = all_plot_data[frame, 0][::vector_subsample, ::vector_subsample]
        v_vec_data = all_plot_data[frame, 1][::vector_subsample, ::vector_subsample]
        quiver.set_UVC(u_vec_data, v_vec_data)
        
        return field_ims + [quiver]
    
    ani = animation.FuncAnimation(fig, update_frame, frames=nt, interval=interval, 
                                  blit=False, repeat=True) # blit=False is often more robust
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for suptitle
    return {'animation': ani, 'figure': fig}


def vis_mechanical_moisture_profiles(model, scale: Scale, 
                                     times_relative=[0, 0.25, 0.5, 0.75, 1.0], 
                                     num_points_profile=100, **kwargs):
    """
    Creates displacement (u, v) and moisture (θ) profiles at different times along a centerline.
    
    Args:
        model: Trained model.
        scale: Scale object.
        domain_vars: Domain object.
        times_relative: List of relative times (0 to 1) to show profiles.
        num_points_profile: Number of points for the profile line.
    """
    
    x_start, x_end = mechanical_moisture_2d_domain.spatial['x']
    y_start, y_end = mechanical_moisture_2d_domain.spatial['y']
    t_start, t_end = mechanical_moisture_2d_domain.temporal['t']
    
    x_line = np.linspace(x_start, x_end, num_points_profile)
    y_center = (y_start + y_end) / 2.0 # Centerline in y
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(times_relative)))
    
    for i, rel_time in enumerate(times_relative):
        t_actual = t_start + rel_time * (t_end - t_start)
        
        XYT = np.column_stack([x_line, 
                              np.full_like(x_line, y_center), 
                              np.full_like(x_line, t_actual)])
        XYT_scaled = XYT / np.array([scale.L, scale.L, scale.t])
        
        predictions = model.predict(XYT_scaled)
        u_line = predictions[:, 0] * scale.epsilon
        v_line = predictions[:, 1] * scale.epsilon
        theta_line = predictions[:, 2] * scale.theta
        
        time_label = f't = {t_actual:.2e} s'
        ax1.plot(x_line, u_line, color=colors[i], label=time_label, linewidth=2)
        ax2.plot(x_line, v_line, color=colors[i], label=time_label, linewidth=2)
        ax3.plot(x_line, theta_line, color=colors[i], label=time_label, linewidth=2)
    
    ax1.set_xlabel('X Position [m]')
    ax1.set_ylabel('u-displacement [m]')
    ax1.set_title('u-displacement Profiles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('X Position [m]')
    ax2.set_ylabel('v-displacement [m]')
    ax2.set_title('v-displacement Profiles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.set_xlabel('X Position [m]')
    ax3.set_ylabel('Moisture Content (θ) [m³/m³]')
    ax3.set_title('Moisture Content Profiles')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig