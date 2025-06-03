import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from process.thermal_mechanical.scale import Scale
from domain_vars import thermal_2d_domain

# You'll need to define this or import it
# thermal_2d_domain = your_domain_object

def vis_2d(model, interval=600, variable_idx=2, **kwargs):
    """
    Generates an animation of a 2D thermal-mechanical model's prediction changing over time.

    Args:
        model: The trained model object with a predict(XYT) method.
        domain: Domain object with spatial and temporal attributes.
        scale: Scale object for proper scaling.
        interval (int, optional): Delay between frames in milliseconds.
        variable_idx (int, optional): Which variable to visualize (0=u, 1=v, 2=T). Defaults to 2 (Temperature).
    """


    var_names = ['u (displacement x)', 'v (displacement y)', 'T (temperature)']
    title = kwargs.get('title', f'Thermal-Mechanical 2D - {var_names[variable_idx]}')
    scale = Scale(thermal_2d_domain)
    
    x_start, x_end = thermal_2d_domain.spatial['x']
    y_start, y_end = thermal_2d_domain.spatial['y']
    t_start, t_end = thermal_2d_domain.temporal['t']
    
    num_x_points = kwargs.get('num_x_points', 50)
    num_y_points = kwargs.get('num_y_points', 50)
    num_t_points = kwargs.get('num_t_points', 50)

    x_points = np.linspace(x_start, x_end, num_x_points)
    y_points = np.linspace(y_start, y_end, num_y_points)
    t_points = np.linspace(t_start, t_end, num_t_points)

    # Create meshgrid for each time step
    X, Y = np.meshgrid(x_points, y_points)
    
    # Prepare data for all time steps
    all_data = []
    
    for t in t_points:
        # Create XYT array for this time step
        T_grid = np.full_like(X, t)
        XYT = np.vstack((X.ravel(), Y.ravel(), T_grid.ravel())).T
        
        # Apply scaling
        XYT_scaled = XYT.copy()
        XYT_scaled[:, 0] *= scale.Lx  # scale x
        XYT_scaled[:, 1] *= scale.Ly  # scale y  
        XYT_scaled[:, 2] *= scale.T   # scale t
        
        # Get predictions
        predictions = model.predict(XYT_scaled)
        
        # Extract the desired variable and reshape
        if predictions.ndim > 1:
            data = predictions[:, variable_idx].reshape(X.shape)
        else:
            data = predictions.reshape(X.shape)
            
        # Apply inverse scaling for visualization
        if variable_idx == 2:  # Temperature
            data = data / scale.Temperature
            
        all_data.append(data)
    
    all_data = np.array(all_data)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Find global min/max for consistent color scale
    vmin, vmax = all_data.min(), all_data.max()
    
    # Create initial plot
    im = ax.imshow(all_data[0], extent=[x_start, x_end, y_start, y_end], 
                   origin='lower', aspect='auto', vmin=vmin, vmax=vmax, 
                   cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(var_names[variable_idx])
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    def update(frame):
        im.set_array(all_data[frame])
        ax.set_title(f'{title} (t={t_points[frame]:.3f})')
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=num_t_points,
                                  interval=interval, blit=True, repeat=True)
    
    # Uncomment to save animation
    # ani.save('thermal_mechanical_animation.mp4', writer='ffmpeg', fps=1000/interval)
    
    return {'field': ani, 'figure': fig}

def vis_2d_all_variables(model, interval=600, **kwargs):
    """
    Creates animations for all three variables (u, v, T) in subplots.
    """
    var_names = ['u (displacement x)', 'v (displacement y)', 'T (temperature)']
    scale = Scale(thermal_2d_domain)
    
    x_start, x_end = thermal_2d_domain.spatial['x']
    y_start, y_end = thermal_2d_domain.spatial['y']
    t_start, t_end = thermal_2d_domain.temporal['t']

    num_x_points = kwargs.get('num_x_points', 50)
    num_y_points = kwargs.get('num_y_points', 50)
    num_t_points = kwargs.get('num_t_points', 50)

    x_points = np.linspace(x_start, x_end, num_x_points)
    y_points = np.linspace(y_start, y_end, num_y_points)
    t_points = np.linspace(t_start, t_end, num_t_points)

    X, Y = np.meshgrid(x_points, y_points)
    
    # Prepare data for all variables and time steps
    all_data = {0: [], 1: [], 2: []}  # u, v, T
    
    for t in t_points:
        T_grid = np.full_like(X, t)
        XYT = np.vstack((X.ravel(), Y.ravel(), T_grid.ravel())).T
        
        XYT_scaled = XYT.copy()
        XYT_scaled[:, 0] /= scale.Lx
        XYT_scaled[:, 1] /= scale.Ly
        XYT_scaled[:, 2] /= scale.t
        
        predictions = model.predict(XYT_scaled)
        
        switch = [scale.Lx, scale.Ly, scale.Temperature]
        for var_idx in range(3):
            data = predictions[:, var_idx].reshape(X.shape)
            data = data * switch[var_idx]
            all_data[var_idx].append(data)
    
    for var_idx in range(3):
        all_data[var_idx] = np.array(all_data[var_idx])
    
    # Set up subplot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ims = []
    cbars = []
    
    for var_idx in range(3):
        vmin, vmax = all_data[var_idx].min(), all_data[var_idx].max()
        im = axes[var_idx].imshow(all_data[var_idx][0], 
                                 extent=[x_start, x_end, y_start, y_end],
                                 origin='lower', aspect='auto', 
                                 vmin=vmin, vmax=vmax, cmap='viridis')
        ims.append(im)
        
        cbar = plt.colorbar(im, ax=axes[var_idx])
        cbar.set_label(var_names[var_idx])
        cbars.append(cbar)
        
        axes[var_idx].set_xlabel('x')
        axes[var_idx].set_ylabel('y')
        axes[var_idx].set_title(var_names[var_idx])
    
    def update(frame):
        for var_idx in range(3):
            ims[var_idx].set_array(all_data[var_idx][frame])
        fig.suptitle(f'Thermal-Mechanical 2D (t={(t_points[frame]/(60*60*24)):.3f}) days)')
        return ims

    ani = animation.FuncAnimation(fig, update, frames=num_t_points,
                                  interval=interval, blit=True, repeat=True)
    
    return {'field': ani, 'figure': fig}