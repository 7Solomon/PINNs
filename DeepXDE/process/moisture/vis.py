from FEM.output import load_fem_results, save_fem_results

from process.moisture.gnd import get_richards_1d_head_fem, get_richards_1d_saturation_fem
from utils.metadata import Domain
from domain_vars import moisture_1d_domain, moisture_2d_domain
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from process.moisture.scale import *

from mpi4py import MPI



def vis_1d_head(model, scale: HeadScale, interval=2000, xlabel='z', ylabel='u(z,t)', **kwargs):
    title= f'Richards 1d' if 'title' not in kwargs else kwargs['title']
    z_start, z_end = moisture_1d_domain.spatial['z']
    t_start, t_end = moisture_1d_domain.temporal['t']
    num_z_points = 25
    num_t_points = 20

    z_points = np.linspace(z_start, z_end, num_z_points)
    t_points = np.linspace(t_start, t_end, num_t_points)

    Z, T = np.meshgrid(z_points, t_points)

    Z_scaled = Z.copy() / scale.L   # Scale z
    T_scaled = T.copy() / scale.T   # Scale t
    ZT_scaled = np.vstack((Z_scaled.ravel(), T_scaled.ravel())).T


    #ground_eval_flat_time_spatial = load_fem_results("BASELINE/moisture/1d/ground_truth.npy")
    _, ground_eval_flat_time_spatial = get_richards_1d_head_fem(
            moisture_1d_domain,
            nz=num_z_points,
            evaluation_times=t_points,
            evaluation_spatial_points_z= z_points.reshape(-1, 1),
        )
    save_fem_results("BASELINE/moisture/1d/ground_truth.npy", ground_eval_flat_time_spatialyy)

    ground_truth_data = np.full((num_z_points, num_t_points), np.nan) 
    if ground_eval_flat_time_spatial is not None and ground_eval_flat_time_spatial.size > 0:
        try:
            # Reshape the flattened [time, space] data into [nt, nz]
            # then transpose to get [nz, nt] for plotting consistency.
            ground_truth_data = ground_eval_flat_time_spatial.reshape(num_t_points, num_z_points).transpose(1, 0)
        except ValueError as e:
            print(f"Error reshaping FEM results: {e}")
            print(f"ground_eval_flat_time_spatial shape: {ground_eval_flat_time_spatial.shape}")
            print(f"Target reshape dimensions: ({num_t_points}, {num_z_points})")
    else:
        print("Warning: FEM evaluation data is None or empty on rank 0. Plotting with NaNs.")


    # Get predictions from the model
    predictions = model.predict(ZT_scaled)
    predictions = predictions * scale.H
    pinn_data = predictions.reshape(num_t_points, num_z_points).T

    error_data = pinn_data - ground_truth_data
    #mse = np.mean(np.square(error_data), axis=0)

    # --- Setup Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    

    # --- Create Lines ---
    line_pinn, = axes[0].plot(z_points, pinn_data[:, 0], 'b-')
    line_gt, = axes[1].plot(z_points, ground_truth_data[:, 0], 'r-')
    line_err, = axes[2].plot(z_points, error_data[:, 0], 'g-')

    #abs_error_T = np.abs(error_data).T # Transpose for plotting
    #contour = axes[3].contourf(z_points, t_points, abs_error_T, cmap='Reds', levels=20)
    #fig.colorbar(contour, ax=axes[3], label='Absolute Error Magnitude')
    #time_indicator = axes[3].axhline(y=t_points[0], color='black', linestyle='--')


    # --- Formatting ---
    axes[0].set(title='PINN Prediction', xlabel=xlabel, ylabel=ylabel, xlim=(z_start, z_end))
    axes[1].set(title='Ground Truth', xlabel=xlabel, xlim=(z_start, z_end))
    axes[2].set(title='Absolute Error', xlabel=xlabel, xlim=(z_start, z_end))
    #axes[3].set(title='Error Contour (Z-T Plane)', xlabel=xlabel, ylabel='Time [s]')
    #axes[3].sharey = False 
    
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
                                  interval=interval, blit=True, repeat=False)
    
    #ani.save('animation.mp4', writer='ffmpeg', fps=1000/interval)
    #plt.show() # geht glaube nocht auf ssh
    return {'field': ani}


def vis_1d_saturation(model, scale: SaturationScale, interval=200, xlabel='z [m]', ylabel='Saturation [-]', **kwargs):
    title = 'Richards 1D (Saturation)' if 'title' not in kwargs else kwargs['title']
    z_start, z_end = moisture_1d_domain.spatial['z']
    t_start, t_end = moisture_1d_domain.temporal['t']
    num_z_points = 50
    num_t_points = 50

    z_points = np.linspace(z_start, z_end, num_z_points)
    t_points = np.linspace(t_start, t_end, num_t_points)

    # --- Get Ground Truth Data ---
    #_, ground_eval_flat_time_spatial = get_richards_1d_saturation_fem(
    #    moisture_1d_domain,
    #    nz=num_z_points,
    #    evaluation_times=t_points,
    #    evaluation_spatial_points_z=z_points.reshape(-1, 1),
    #)
    #save_fem_results("BASELINE/moisture/1d/saturation_ground_truth.npy", ground_eval_flat_time_spatial)
    ground_eval_flat_time_spatial = load_fem_results("BASELINE/moisture/1d/saturation_ground_truth.npy")

    ground_truth_data = np.full((num_z_points, num_t_points), np.nan)
    if ground_eval_flat_time_spatial is not None and ground_eval_flat_time_spatial.size > 0:
        try:
            ground_truth_data = ground_eval_flat_time_spatial.reshape(num_t_points, num_z_points).T
        except ValueError as e:
            print(f"Error reshaping FEM results: {e}")
    else:
        print("Warning: FEM evaluation data is None or empty. Ground truth will be NaNs.")

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
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # --- Create Lines ---
    line_pinn, = axes[0].plot(z_points, pinn_data[:, 0], 'b-')
    line_gt, = axes[1].plot(z_points, ground_truth_data[:, 0], 'r-')
    line_err, = axes[2].plot(z_points, error_data[:, 0], 'g-')

    # --- Formatting ---
    axes[0].set(title='PINN Prediction', xlabel=xlabel, ylabel=ylabel, xlim=(z_start, z_end))
    axes[1].set(title='Ground Truth', xlabel=xlabel, xlim=(z_start, z_end))
    axes[2].set(title='Absolute Error', xlabel=xlabel, xlim=(z_start, z_end))


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
                                  interval=interval, blit=True, repeat=False)

    return {'field': ani}

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

