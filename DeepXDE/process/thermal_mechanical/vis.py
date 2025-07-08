import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from FEM.output import load_fem_results, save_fem_results
from process.thermal_mechanical.scale import Scale
from domain_vars import thermal_mechanical_2d_domain
from process.thermal_mechanical.gnd import get_thermal_mechanical_fem

def vis_2d_multi(model, scale: Scale, points_data:dict, fem_value_points, interval=200, **kwargs):

    # --- Configuration ---
    var_names = ['U (X-Disp)', 'V (Y-Disp)', 'Temperature', 'Disp. Magnitude']
    plot_titles_comp = ['PINN Prediction', 'Ground Truth (FEM)', 'Absolute Error']
    cmaps = ['RdBu_r', 'RdBu_r', 'plasma', 'viridis']
    
    # --- Domain and Grid Setup ---
    x_start, x_end = thermal_mechanical_2d_domain.spatial['x']
    y_start, y_end = thermal_mechanical_2d_domain.spatial['y']
    t_points = points_data['temporal_coords']['t']  # Extract time points
    
    nx, ny, nt = points_data['resolution']['x'], points_data['resolution']['y'], points_data['resolution']['t']

    gt_u, gt_v, gt_T = fem_value_points[:, :, :, 0], fem_value_points[:, :, :, 1], fem_value_points[:, :, :, 2]
    gt_mag = np.sqrt(gt_u**2 + gt_v**2)

    # --- 2. Get PINN Predictions ---
    print("Generating PINN predictions for all time steps...")
    pinn_data = np.zeros((nt, ny, nx, 3))
    spacetime_points = points_data['spacetime_points_flat']
    XYT_scaled = spacetime_points / np.array([scale.L, scale.L, scale.t])    
    preds_scaled = model.predict(XYT_scaled)
    
    preds_scaled[:, 0] *= scale.U  # Scale u
    preds_scaled[:, 1] *= scale.U  # Scale v  
    preds_scaled[:, 2] *= scale.Temperature  # Scale T
    
    # Reshape to (nt, ny, nx, 3)
    pinn_data = points_data['reshape_utils']['pred_to_ij'](preds_scaled)
    pinn_u, pinn_v, pinn_T = pinn_data[:, :, :, 0], pinn_data[:, :, :, 1], pinn_data[:, :, :, 2]
    pinn_mag = np.sqrt(pinn_u**2 + pinn_v**2)

    # --- 3. Calculate Error Data ---
    error_u, error_v, error_T = pinn_u - gt_u, pinn_v - gt_v, pinn_T - gt_T
    error_mag = pinn_mag - gt_mag

    # --- Data Lists for easier looping ---
    pinn_all = [pinn_u, pinn_v, pinn_T, pinn_mag]
    comp_all = [
        [pinn_u, gt_u, error_u], [pinn_v, gt_v, error_v],
        [pinn_T, gt_T, error_T], [pinn_mag, gt_mag, error_mag]
    ]

    # --- 4. PLOT 1: PINN-only Visualization (1x4) ---
    fig1, axes1 = plt.subplots(1, 4, figsize=(22, 5))
    ims1 = []
    for i in range(4): # Loop over variables
        ax = axes1[i]
        data_slice = pinn_all[i][0] # First time step
        vmin, vmax = np.min(pinn_all[i]), np.max(pinn_all[i])
        im = ax.imshow(data_slice, vmin=vmin, vmax=vmax, cmap=cmaps[i], origin='lower', extent=[x_start, x_end, y_start, y_end])
        ims1.append(im)
        fig1.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"PINN: {var_names[i]}")
        ax.set_xlabel("x [m]"), ax.set_ylabel("y [m]")

    def update1(frame):
        for i in range(4):
            ims1[i].set_data(pinn_all[i][frame])
        fig1.suptitle(f'PINN Prediction (Time: {t_points[frame]:.2f} s)', fontsize=16)
        return ims1

    ani1 = animation.FuncAnimation(fig1, update1, frames=nt, interval=interval, blit=False, repeat=True)
    fig1.tight_layout(rect=[0, 0, 1, 0.95])

    # --- 5. PLOT 2: Comparison Visualization (4x3) ---
    fig2, axes2 = plt.subplots(4, 3, figsize=(15, 18))
    ims2 = []
    for i in range(4):  # Loop over variables
        vmin_pinn_gt = min(np.min(comp_all[i][0]), np.min(comp_all[i][1]))
        vmax_pinn_gt = max(np.max(comp_all[i][0]), np.max(comp_all[i][1]))
        vmax_error = np.max(np.abs(comp_all[i][2]))

        for j in range(3):  # Loop over columns (PINN, GT, Error)
            ax = axes2[i, j]
            data_slice = comp_all[i][j][0]
            
            if j < 2: # PINN or GT plots
                im = ax.imshow(data_slice, vmin=vmin_pinn_gt, vmax=vmax_pinn_gt, cmap=cmaps[i], origin='lower', extent=[x_start, x_end, y_start, y_end])
            else: # Error plot
                im = ax.imshow(data_slice, vmin=-vmax_error, vmax=vmax_error, cmap='bwr', origin='lower', extent=[x_start, x_end, y_start, y_end])
            
            ims2.append(im)
            fig2.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(f"{var_names[i]}: {plot_titles_comp[j]}")
            ax.set_xlabel("x [m]"), ax.set_ylabel("y [m]")

    def update2(frame):
        k = 0
        for i in range(4):
            for j in range(3):
                ims2[k].set_data(comp_all[i][j][frame])
                k += 1
        fig2.suptitle(f'PINN vs. FEM Comparison (Time: {t_points[frame]:.2f} s)', fontsize=16)
        return ims2

    ani2 = animation.FuncAnimation(fig2, update2, frames=nt, interval=interval, blit=False, repeat=True)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    return {'field': ani2, 'fig': fig2}