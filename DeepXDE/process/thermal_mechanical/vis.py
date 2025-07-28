import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from points import get_meshgrid_for_visualization
from FEM.output import load_fem_results, save_fem_results
from process.thermal_mechanical.scale import Scale
from domain_vars import thermal_mechanical_2d_domain
from process.thermal_mechanical.gnd import get_thermal_mechanical_fem

def get_pinn_data(model, points_data: dict, scale):
    # 2. PINN Prediction Data
    spacetime_points = points_data['spacetime_points_flat']
    print(f"Spacetime points shape: {spacetime_points.shape}")
    XYT_scaled = spacetime_points / np.array([scale.L, scale.L, scale.t])
    
    print("Running model predictions...")
    preds_scaled = model.predict(XYT_scaled)
    print(f"Raw predictions shape: {preds_scaled.shape}")

    # Rescale predictions to original physical units
    preds_scaled[:, 0] *= scale.U
    preds_scaled[:, 1] *= scale.U
    preds_scaled[:, 2] *= scale.Temperature
    
    # Reshape flat predictions to a grid using the provided utility
    pinn_data = points_data['reshape_utils']['pred_to_ij'](preds_scaled)
    print(f"Reshaped PINN data shape: {pinn_data.shape}")
    return pinn_data

def _prepare_plot_data(points_data: dict, fem_value_points, pinn_data):
    """
    Helper function to generate PINN, Ground Truth (FEM), and error data.
    This centralizes data processing to be used by multiple visualization functions.
    """
    print("Preparing visualization data...")
    print(f"Original fem_value_points shape: {fem_value_points.shape}")
    
    # Get expected dimensions from resolution
    nx = points_data['resolution']['x']  # 10
    ny = points_data['resolution']['y']  # 100  
    nt = points_data['resolution']['t']  # 50
    print(f"Expected dimensions: nx={nx}, ny={ny}, nt={nt}")
    
    # 1. Ground Truth (FEM) Data - need to understand the actual structure
    if fem_value_points.shape == (nt, ny, nx, 3):
        print("FEM data is (nt, ny, nx, n_vars) - need to transpose")
        # If FEM data is (nt, ny, nx, n_vars), we need to reshape to (nx, ny, nt)
        gt_u = fem_value_points[..., 0].transpose(2, 1, 0)  # (nt, ny, nx) -> (nx, ny, nt)
        gt_v = fem_value_points[..., 1].transpose(2, 1, 0)
        gt_T = fem_value_points[..., 2].transpose(2, 1, 0)
    elif fem_value_points.shape == (nt, nx, ny, 3):
        print("FEM data is (nt, nx, ny, n_vars) - need to transpose")
        gt_u = fem_value_points[..., 0].transpose(1, 2, 0)  # (nt, nx, ny) -> (nx, ny, nt)
        gt_v = fem_value_points[..., 1].transpose(1, 2, 0)
        gt_T = fem_value_points[..., 2].transpose(1, 2, 0)
    elif fem_value_points.shape == (nx, ny, nt, 3):
        print("FEM data is already (nx, ny, nt, n_vars)")
        gt_u = fem_value_points[..., 0]
        gt_v = fem_value_points[..., 1]
        gt_T = fem_value_points[..., 2]
    else:
        raise ValueError(f"Unexpected fem_value_points shape: {fem_value_points.shape}")
    
    gt_mag = np.sqrt(gt_u**2 + gt_v**2)
    print(f"GT data shapes after processing: U={gt_u.shape}, V={gt_v.shape}, T={gt_T.shape}")


    pinn_u = pinn_data[..., 0]
    pinn_v = pinn_data[..., 1]
    pinn_T = pinn_data[..., 2]
    pinn_mag = np.sqrt(pinn_u**2 + pinn_v**2)
    print(f"PINN data shapes: U={pinn_u.shape}, V={pinn_v.shape}, T={pinn_T.shape}")

    # 3. Error Data
    error_u = pinn_u - gt_u
    error_v = pinn_v - gt_v
    error_T = pinn_T - gt_T
    error_mag = pinn_mag - gt_mag
    
    # 4. Package for easy access
    data_package = {
        'pinn': [pinn_u, pinn_v, pinn_T, pinn_mag],
        'gt': [gt_u, gt_v, gt_T, gt_mag],
        'error': [error_u, error_v, error_T, error_mag]
    }
    print("Data preparation complete.")
    return data_package
def create_comparison_gif(
    data: dict, 
    domain_info: dict, 
    time_points: np.ndarray,
    interval: int = 200, 
    fps: int = 10
):
    """
    Creates a 4x3 comparison animation (PINN vs. GT vs. Error).
    Also returns static figures for each subplot at the final time step.
    Assumes data format is (nx, ny, nt).
    """
    var_names = ['U (X-Disp)', 'V (Y-Disp)', 'Temperature', 'Disp. Magnitude']
    plot_titles = ['PINN Prediction', 'Ground Truth (FEM)', 'Absolute Error']
    cmaps = ['RdBu_r', 'RdBu_r', 'plasma', 'viridis']
    
    pinn_all, gt_all, error_all = data['pinn'], data['gt'], data['error']
    comp_all = [list(z) for z in zip(pinn_all, gt_all, error_all)]
    
    nt = len(time_points)
    x_start, x_end = domain_info['x']
    y_start, y_end = domain_info['y']

    fig, axes = plt.subplots(4, 3, figsize=(15, 18), constrained_layout=True)
    ims = []

    # Store static figures for each subplot at final time step
    end_figs = {}

    for i in range(4):  # Loop over variables
        vmin_pinn_gt = min(np.min(comp_all[i][0]), np.min(comp_all[i][1]))
        vmax_pinn_gt = max(np.max(comp_all[i][0]), np.max(comp_all[i][1]))
        vmax_error = np.max(np.abs(comp_all[i][2]))
        
        for j in range(3):  # Loop over columns (PINN, GT, Error)
            ax = axes[i, j]
            # Data should be (nx, ny, nt), so we want [:, :, 0] for first time step
            data_slice = comp_all[i][j][:, :, 0].T
            if j < 2:
                cmap, vmin, vmax = cmaps[i], vmin_pinn_gt, vmax_pinn_gt
            else:
                cmap, vmin, vmax = 'bwr', -vmax_error, vmax_error
            
            im = ax.imshow(data_slice, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower', 
                          extent=[x_start, x_end, y_start, y_end], interpolation='bicubic')
            ims.append(im)
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(f"{var_names[i]}: {plot_titles[j]}")
            ax.set_xlabel("x [m]"), ax.set_ylabel("y [m]")

    # Animation update function
    def update(frame):
        k = 0
        for i in range(4):
            for j in range(3):
                frame_data = comp_all[i][j][:, :, frame].T
                ims[k].set_data(frame_data)
                k += 1
        fig.suptitle(f'PINN vs. FEM Comparison (Time: {time_points[frame]:.2f} s)', fontsize=16)
        return ims

    ani = animation.FuncAnimation(fig, update, frames=nt, interval=interval, blit=False, repeat=True)

    # Create static figures for each subplot at final time step
    for i in range(4):
        for j in range(3):
            sub_fig, sub_ax = plt.subplots(figsize=(4, 12))  # removed constrained_layout
            data_slice = comp_all[i][j][:, :, -1].T  # final time step
            if j < 2:
                cmap, vmin, vmax = cmaps[i], np.min(data_slice), np.max(data_slice)
            else:
                cmap, vmin, vmax = 'bwr', -np.max(np.abs(data_slice)), np.max(np.abs(data_slice))
            im = sub_ax.imshow(data_slice, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower', 
                            extent=[x_start, x_end, y_start, y_end], interpolation='bicubic')
            sub_fig.colorbar(im, ax=sub_ax, shrink=0.8, pad=0.1)
            # sub_ax.set_title(f"{var_names[i]}: {plot_titles[j]} (Final t)")  # <-- removed title
            sub_ax.set_xlabel("x [m]")
            sub_ax.set_ylabel("y [m]")
            plt.tight_layout() 
            end_figs[f"{var_names[i]}_{plot_titles[j]}"] = sub_fig
    return {'gif': ani, 'fig': fig, **end_figs}

def plot_probe_point_evaluations(
    data: dict, 
    points_data: dict,
    probe_points: list, 
):
    """
    Generates time-series plots for variables at specified probe points.
    Assumes data format is (nx, ny, nt).
    """
    print("Generating probe point evaluations...")
    var_names = ['U (X-Disp)', 'V (Y-Disp)', 'Temperature']
    pinn_all, gt_all = data['pinn'], data['gt']
    
    # Get coordinate vectors from points_data
    t_coords = points_data['temporal_coords']['t']
    x_coords = points_data['spatial_coords']['x']
    y_coords = points_data['spatial_coords']['y']

    figures = {}
    
    for i, (x_p, y_p) in enumerate(probe_points):
        # Find the nearest grid indices for the probe point
        ix = np.argmin(np.abs(x_coords - x_p))
        iy = np.argmin(np.abs(y_coords - y_p))

        fig, axes = plt.subplots(1, 3, figsize=(21, 5), constrained_layout=True)
        fig.suptitle(f'Time-Series Analysis at Probe Point ({x_coords[ix]:.3f}, {y_coords[iy]:.3f})m', fontsize=16)

        for j in range(3):  # Loop over U, V, T
            ax = axes[j]
            
            # Data is (nx, ny, nt), so extract time series at [ix, iy, :]
            pinn_ts = pinn_all[j][ix, iy, :]
            gt_ts = gt_all[j][ix, iy, :]

            ax.plot(t_coords, gt_ts, 'k-', label='Ground Truth (FEM)', linewidth=2.5)
            ax.plot(t_coords, pinn_ts, 'r--', label='PINN Prediction', linewidth=2)
            ax.set_title(var_names[j])
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, linestyle=':')
        
        figures[f'probe_eval_{i}'] = fig

    return figures
def vis_thermo_mechanical_2d(model, scale: Scale, points_data: dict, fem_value_points):
    pinn_data = get_pinn_data(model, points_data, scale)
    data = _prepare_plot_data(points_data, fem_value_points, pinn_data)
    domain_info = {
        'x': thermal_mechanical_2d_domain.spatial['x'],
        'y': thermal_mechanical_2d_domain.spatial['y']
    }
    time_points = points_data['temporal_coords']['t']
    gif = create_comparison_gif(
        data=data,
        domain_info=domain_info,
        time_points=time_points
    )

    prob_locations = [
        (0.05, 0.5),      # A point in the middle
        (0.1, 1.0),       # A point at the top right boundary
        (0.0, 0.0)        # A point at the bottom left boundary
    ]
    probe_fig = plot_probe_point_evaluations(
        data=data,
        points_data=points_data,
        probe_points=prob_locations
    )
    return {
        **gif,
        **probe_fig,
    }

def vis_thermo_callback(pinn_data, points_data: dict, fem_value_points):
    data = _prepare_plot_data(points_data, fem_value_points, pinn_data)
    domain_info = {
        'x': thermal_mechanical_2d_domain.spatial['x'],
        'y': thermal_mechanical_2d_domain.spatial['y']
    }
    time_points = points_data['temporal_coords']['t']
    gif = create_comparison_gif(
        data=data,
        domain_info=domain_info,
        time_points=time_points
    )

    prob_locations = [
        (0.05, 0.5),      # A point in the middle
        (0.1, 1.0),       # A point at the top right boundary
        (0.0, 0.0)        # A point at the bottom left boundary
    ]
    probe_fig = plot_probe_point_evaluations(
        data=data,
        points_data=points_data,
        probe_points=prob_locations
    )
    return {
        **gif,
        **probe_fig,
    }