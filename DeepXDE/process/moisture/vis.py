from vis import get_2d_domain
from utils.metadata import Domain
from domain_vars import moisture_1d_domain, moisture_2d_domain
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from process.moisture.scale import Scale

from vis import get_2d_time_domain

def vis_1d_head(model, interval=2000, xlabel='z', ylabel='u(z,t)', **kwargs):
    """
    Generates an animation of a 1D model's prediction changing over time.

    Args:
        model: The trained model object with a predict(XT) method.
        spatial_domain (tuple): (x_start, x_end, num_x_points) for the spatial domain.
        time_domain (tuple): (t_start, t_end, num_t_points) for the time domain.
        interval (int, optional): Delay between frames in milliseconds. Defaults to 2000.
        xlabel (str, optional): Label for the x-axis. Defaults to 'x'.
        ylabel (str, optional): Label for the y-axis. Defaults to 'u(x,t)'.
    """

    title= f'Richards 1d' if 'title' not in kwargs else kwargs['title']
    z_start, z_end = moisture_1d_domain.spatial['z']
    t_start, t_end = moisture_1d_domain.temporal['t']
    num_x_points = 100
    num_t_points = 100
    scale = Scale(moisture_1d_domain)

    z_points = np.linspace(z_start, z_end, num_x_points)
    t_points = np.linspace(t_start, t_end, num_t_points)

    Z, T = np.meshgrid(z_points, t_points)

    Z_scaled = Z.copy() / scale.L   # Scale z
    T_scaled = T.copy() / scale.T   # Scale t
    ZT_scaled = np.vstack((Z_scaled.ravel(), T_scaled.ravel())).T


    # Get predictions from the model
    predictions = model.predict(ZT_scaled)
    predictions = predictions * scale.H
    # 
    data = predictions.reshape(num_t_points, num_x_points).T


    
    fig, ax = plt.subplots()
    line, = ax.plot(z_points, data[:, 0])

    ax.set_xlim(z_points.min(), z_points.max())
    ax.set_ylim(data.min(), data.max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    time_scale = ((60*60), 'Hours') if t_points.max()/(60*60*24) < 2 else ((60*60*24), 'Days')

    def update(frame):
        line.set_ydata(data[:, frame])
        ax.set_title(f'{title} (t={(t_points[frame]/time_scale[0]):.3f} {time_scale[1]})')
        return line,


    ani = animation.FuncAnimation(fig, update, frames=num_t_points,
                                  interval=interval, blit=True, repeat=False)
    
    #ani.save('animation.mp4', writer='ffmpeg', fps=1000/interval)
    #plt.show() # geht glaube nocht auf ssh
    return {'field': ani}

def vis_1d_saturation(model, interval=2000, xlabel='z', ylabel='u(z,t)', **kwargs):

    title= f'Richards 1d' if 'title' not in kwargs else kwargs['title']
    z_start, z_end = moisture_1d_domain.spatial['z']
    t_start, t_end = moisture_1d_domain.temporal['t']
    num_x_points = 100
    num_t_points = 100
    scale = Scale(moisture_1d_domain)

    z_points = np.linspace(z_start, z_end, num_x_points)
    t_points = np.linspace(t_start, t_end, num_t_points)

    Z, T = np.meshgrid(z_points, t_points)
    ZT = np.vstack((Z.ravel(), T.ravel())).T

    Z_scaled = Z.copy() / scale.L  # Scale z
    T_scaled = T.copy() / scale.T  # Scale t
    ZT_scaled = np.vstack((Z_scaled.ravel(), T_scaled.ravel())).T


    # Get predictions from the model
    predictions = model.predict(ZT_scaled)


    if predictions.ndim > 1 and predictions.shape[1] > 1:
        print(f"Warning: Model output has shape {predictions.shape}. Assuming the first column is the desired output.")

    try:
        data = predictions.reshape(num_t_points, num_x_points).T
    except ValueError as e:
        print(f"Error reshaping predictions: {e}")
        print(f"Prediction shape: {predictions.shape}, Target shape: ({num_t_points}, {num_x_points}) then transposed.")
        print("Ensure the total number of elements in predictions matches num_x_points * num_t_points.")
        return None


    fig, ax = plt.subplots()
    line, = ax.plot(z_points, data[:, 0])

    ax.set_xlim(z_points.min(), z_points.max())
    ax.set_ylim(data.min(), data.max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    time_scale = ((60*60), 'Hours') if t_points.max()/(60*60*24) < 2 else ((60*60*24), 'Days')

    def update(frame):
        line.set_ydata(data[:, frame])
        ax.set_title(f'{title} (t={(t_points[frame]/time_scale[0]):.3f} {time_scale[1]})')
        return line,

    ani = animation.FuncAnimation(fig, update, frames=num_t_points,
                                  interval=interval, blit=True, repeat=False)
    
    #ani.save('animation.mp4', writer='ffmpeg', fps=1000/interval)
    #plt.show() # geht glaube nocht auf ssh
    return {'field': ani}


def visualize_2d_mixed(model, **kwargs):

    scale = Scale(moisture_2d_domain)
    domain = get_2d_domain(moisture_2d_domain, scale)
    points, X, Y, nx, ny = domain['normal']
    scaled_points, X_scaled, Y_scaled, _, _ = domain['scaled']

    # Get predictions
    predictions = model.predict(scaled_points)
    
    # Extract head (index 0) and saturation (index 1)
    head_predictions = predictions[:, 0] * scale.H  # Rescale head to physical units
    saturation_predictions = predictions[:, 1]  # Saturation is typically dimensionless [0,1]

    # Create subplots for both fields
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot head field
    head_contour = ax1.contourf(X.reshape(ny, nx), Y.reshape(ny, nx), 
                               head_predictions.reshape(ny, nx), levels=50, cmap='Blues')
    head_cbar = fig.colorbar(head_contour, ax=ax1)
    head_cbar.set_label('Head [m]')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('Head Field')
    
    # Plot saturation field
    sat_contour = ax2.contourf(X.reshape(ny, nx), Y.reshape(ny, nx), 
                              saturation_predictions.reshape(ny, nx), levels=50, cmap='Reds')
    sat_cbar = fig.colorbar(sat_contour, ax=ax2)
    sat_cbar.set_label('Saturation [-]')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_title('Saturation Field')
    
    # Overall title
    title = '2D Richards Equation: Head and Saturation' if 'title' not in kwargs else kwargs['title']
    fig.suptitle(title)
    
    plt.tight_layout()
    
    return {'head_saturation': fig, 'head_field': head_predictions.reshape(ny, nx), 
            'saturation_field': saturation_predictions.reshape(ny, nx)}


def visualize_2d_darcy(model, **kwargs):
    scale = Scale(moisture_2d_domain)
    domain = get_2d_domain(moisture_2d_domain, scale)
    points, X, Y, nx, ny = domain['normal']
    scaled_points, X_scaled, Y_scaled, _, _ = domain['scaled']

    # Get predictions
    predictions = model.predict(scaled_points)
    predictions = predictions * scale.H  # Rescale predictions to physical units

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the field
    contour = ax.contourf(X.reshape(ny, nx), Y.reshape(ny, nx), predictions.reshape(ny, nx), levels=50, cmap='Blues')
    
    cbar = fig.colorbar(contour)
    cbar.set_label('Field Prediction')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('2D Darcy Flow Field Visualization' if 'title' not in kwargs else kwargs['title'])

    return {'field': fig}

