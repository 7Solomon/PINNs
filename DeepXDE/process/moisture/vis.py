from FEM.output import load_fem_results, save_fem_results

from process.moisture.gnd import get_richards_1d_head_fem, get_richards_1d_head_fem_points, get_richards_1d_saturation_fem
from utils.metadata import Domain
from domain_vars import moisture_1d_domain, moisture_2d_domain
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from process.moisture.scale import *

from mpi4py import MPI



def vis_1d_head(model, scale: HeadScale, points_data:dict, ground_truth_data:np.ndarray, interval=1000, xlabel='z', ylabel='u(z,t)', **kwargs):
    title= f'Richards 1d' if 'title' not in kwargs else kwargs['title']
    z_start, z_end = moisture_1d_domain.spatial['z']
    t_start, t_end = moisture_1d_domain.temporal['t']
    
    z_points = points_data['spatial_coords']['z']
    t_points = points_data['temporal_coords']['t']
    num_z_points = points_data['resolution']['z']
    num_t_points = points_data['resolution']['t']

    Z, T = np.meshgrid(z_points, t_points)

    Z_scaled = Z.copy() / scale.L   # Scale z
    T_scaled = T.copy() / scale.T   # Scale t
    ZT_scaled = np.vstack((Z_scaled.ravel(), T_scaled.ravel())).T


    # Get predictions from the model
    predictions = model.predict(ZT_scaled)
    predictions = predictions * scale.h_char
    pinn_data = predictions.reshape(num_t_points, num_z_points).T

    error_data = pinn_data - ground_truth_data
    #mse = np.mean(np.square(error_data), axis=0)

    # --- Setup Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    

    # --- Create Lines ---
    line_pinn, = axes[0].plot(z_points, pinn_data[:, 0], 'b-')
    line_gt, = axes[1].plot(z_points, ground_truth_data[:, 0], 'r-')
    line_err, = axes[2].plot(z_points, error_data[:, 0], 'g-')

    #abs_error_T = np.abs(error_data).T # Transpose for plotting
    #contour = axes[3].contourf(z_points, t_points, abs_error_T, cmap='Reds', levels=20)
    #fig.colorbar(contour, ax=axes[3], label='Absolute Error Magnitude')
    #time_indicator = axes[3].axhline(y=t_points[0], color='black', linestyle='--')


    # --- Formatting ---
    axes[0].set(title='PINN Prediction', xlabel=xlabel, ylabel=ylabel, xlim=(z_start, z_end), ylim=(0,-10))
    axes[1].set(title='Ground Truth', xlabel=xlabel, ylabel=ylabel, xlim=(z_start, z_end), ylim=(0,-10))
    axes[2].set(title='Absolute Error', xlabel=xlabel, ylabel='Error', xlim=(z_start, z_end))
    
    # For error
    error_min, error_max = np.nanmin(error_data), np.nanmax(error_data)
    margin = (error_max - error_min) * 0.1  # 10% margin
    axes[2].set_ylim(error_min - margin, error_max + margin)
    axes[2].sharey = False # Unshare the y-axis for the error plot

    error_range_text = f'Global Min: {error_min:.2e}\nGlobal Max: {error_max:.2e}'
    axes[2].text(0.02, 0.98, error_range_text, transform=axes[2].transAxes,
                 fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    time_scale_val, time_scale_unit = ((60*60), 'Hours') if t_end / (60*60*24) < 2 else ((60*60*24), 'Days')
    fig.suptitle(f'{title} (t=0.00 {time_scale_unit})')# | Overall MSE: {mse:.2e}')    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    def update(frame):
        line_pinn.set_ydata(pinn_data[:, frame])
        line_gt.set_ydata(ground_truth_data[:, frame])
        line_err.set_ydata(error_data[:, frame])

        #time_indicator.set_ydata(t_points[frame])

        current_time = t_points[frame] / time_scale_val
        fig.suptitle(f'{title} (t={current_time:.2f} {time_scale_unit})')# | Overall MSE: {mse:.2e}')
        return line_pinn, line_gt, line_err#, time_indicator



    ani = animation.FuncAnimation(fig, update, frames=num_t_points,
                                  interval=interval, repeat=False)
    
    #ani.save('animation.mp4', writer='ffmpeg', fps=1000/interval)
    #plt.show() # geht glaube nocht auf ssh
    return {'field': ani, 'fig': fig}


def vis_1d_saturation(model, scale: SaturationScale, points_data:dict, ground_truth_data:np.ndarray, interval=1000, xlabel='z [m]', ylabel='Saturation [-]', **kwargs):
    title = 'Richards 1D (Saturation)' if 'title' not in kwargs else kwargs['title']
    z_start, z_end = moisture_1d_domain.spatial['z']
    t_start, t_end = moisture_1d_domain.temporal['t']
    
    z_points = points_data['spatial_coords']['z']
    t_points = points_data['temporal_coords']['t']
    num_z_points = points_data['resolution']['z']
    num_t_points = points_data['resolution']['t']


    # --- Get PINN Predictions ---
    Z, T = np.meshgrid(z_points, t_points)
    Z_scaled = Z.copy() / scale.L
    T_scaled = T.copy() / scale.T
    ZT_scaled = np.vstack((Z_scaled.ravel(), T_scaled.ravel())).T

    predictions = model.predict(ZT_scaled)
    # Saturation is dimensionless, so no scaling is applied to the output
    pinn_data = predictions.reshape(num_t_points, num_z_points).T

    # --- Calculate Error ---
    error_data = pinn_data - ground_truth_data

    # --- Setup Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Create Lines ---
    line_pinn, = axes[0].plot(z_points, pinn_data[:, 0], 'b-')
    line_gt, = axes[1].plot(z_points, ground_truth_data[:, 0], 'r-')
    line_err, = axes[2].plot(z_points, error_data[:, 0], 'g-')

    # --- Formatting ---
    axes[0].set(title='PINN Prediction', xlabel=xlabel, ylabel=ylabel, xlim=(z_start, z_end), ylim=(0, 1))
    axes[1].set(title='Ground Truth', xlabel=xlabel, ylabel=ylabel, xlim=(z_start, z_end), ylim=(0, 1))
    axes[2].set(title='Absolute Error', xlabel=xlabel, ylabel='Error', xlim=(z_start, z_end))


    time_scale_val, time_scale_unit = ((60*60), 'Hours') if t_end / (60*60*24) < 2 else ((60*60*24), 'Days')
    fig.suptitle(f'{title} (t=0.00 {time_scale_unit})')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update(frame):
        line_pinn.set_ydata(pinn_data[:, frame])
        line_gt.set_ydata(ground_truth_data[:, frame])
        line_err.set_ydata(error_data[:, frame])

        current_time = t_points[frame] / time_scale_val
        fig.suptitle(f'{title} (t={current_time:.2f} {time_scale_unit})')
        return line_pinn, line_gt, line_err

    ani = animation.FuncAnimation(fig, update, frames=num_t_points,
                                  interval=interval, repeat=False)

    return {'field': ani, 'fig': fig, **vis_1d_time_plot(model, scale, points_data, ground_truth_data, **kwargs)}

def vis_1d_time_plot(model, scale: RichardsScale, points_data: dict, ground_truth_data: np.ndarray, **kwargs):
    """
    Visualizes the 1D transient result as a 2D heatmap (space vs. time).
    """
    # --- 1. Extract Data and Parameters ---
    title = kwargs.get('title', 'Richards 1D (Time-Space Plot)')
    ylabel = kwargs.get('ylabel', 'z [m]')
    
    z_points = points_data['spatial_coords']['z']
    t_points = points_data['temporal_coords']['t']
    num_t_points = points_data['resolution']['t']
    num_z_points = points_data['resolution']['z']

    # --- 2. Get PINN Predictions ---
    #Z, T = np.meshgrid(z_points, t_points)
    Z, T = points_data['spacetime_meshgrid']['z'], points_data['spacetime_meshgrid']['t']
    
    Z_scaled = Z / scale.L
    T_scaled = T / scale.T
    ZT_scaled = np.vstack((Z_scaled.ravel(), T_scaled.ravel())).T

    predictions = model.predict(ZT_scaled)
    
    output_scale = scale.value_scale_list[0]

    pinn_data = (predictions * output_scale).reshape(num_t_points, num_z_points).T

    # --- 3. Calculate Error ---
    error_data = pinn_data - ground_truth_data

    # --- 4. Setup Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle(title)

    # Determine shared color limits for prediction and ground truth
    vmin = min(np.min(pinn_data), np.min(ground_truth_data))
    vmax = max(np.max(pinn_data), np.max(ground_truth_data))

    # Determine time unit for plotting
    time_scale_val, time_scale_unit = (86400, 'Days') if scale.T_domain > 86400 else (3600, 'Hours')
    t_points_scaled = t_points / time_scale_val
    
    # --- 5. Create Plots ---
    # PINN Prediction
    im1 = axes[0].pcolormesh(t_points_scaled, z_points, pinn_data, cmap='viridis', vmin=vmin, vmax=vmax, shading='gouraud')
    axes[0].set_title('PINN Prediction')
    axes[0].set_ylabel(ylabel)
    axes[0].set_xlabel(f'Time [{time_scale_unit}]')
    fig.colorbar(im1, ax=axes[0])

    # Ground Truth
    im2 = axes[1].pcolormesh(t_points_scaled, z_points, ground_truth_data, cmap='viridis', vmin=vmin, vmax=vmax, shading='gouraud')
    axes[1].set_title('Ground Truth (FEM)')
    axes[1].set_xlabel(f'Time [{time_scale_unit}]')
    fig.colorbar(im2, ax=axes[1])

    # Absolute Error
    max_err = np.max(np.abs(error_data))
    im3 = axes[2].pcolormesh(t_points_scaled, z_points, error_data, cmap='coolwarm', vmin=-max_err, vmax=max_err, shading='gouraud')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel(f'Time [{time_scale_unit}]')
    fig.colorbar(im3, ax=axes[2])

    # Invert z-axis to show gravity effect from top to bottom
    for ax in axes:
        ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return {'field_plot': fig}




def visualize_2d_mixed(model, scale, **kwargs):

    raise NotImplementedError("Hier scale stuff noch nicht implementiert und mage nicht domain defininition")

    # Get domain and points
    x_min, x_max = moisture_2d_domain.spatial['x']
    y_min, y_max = moisture_2d_domain.spatial['y']
    nx, ny = 100, 50  # Number of points in x and y directions
    x_points = np.linspace(x_min, x_max, nx)
    y_points = np.linspace(y_min, y_max, ny)

    X, Y = np.meshgrid(x_points, y_points)
    points = np.vstack((X.flatten(), Y.flatten())).T
    scaled_points = points / np.array([scale.L, scale.L])  # Assuming scale has L attribute for length

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

