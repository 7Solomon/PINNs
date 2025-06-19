import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm # For colormaps
from process.mechanical_moisture.scale import Scale
from utils.metadata import Domain # Assuming Domain is in utils.metadata

# Import the FEM solver
from process.mechanical_moisture.gnd import get_coupled_transient_fem

from domain_vars import mechanical_moisture_2d_domain

from mpi4py import MPI # For checking rank if needed, though vis usually runs on rank 0

def visualize_transient_mechanical_moisture_comparison(
    pinn_model, 
    scale: Scale, 
    fem_nx=20, fem_ny=10, fem_dt=None, # FEM discretization params
    vis_nx=50, vis_ny=25, vis_nt=30,   # Visualization grid params
    interval=200,
    **kwargs):
    """
    Visualizes PINN predictions vs. FEM ground truth for mechanical-moisture problem.
    Creates a 3x3 animation: (u_x, u_y, theta) x (PINN, FEM, Difference).
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    x_min, x_max = mechanical_moisture_2d_domain.spatial['x']
    y_min, y_max = mechanical_moisture_2d_domain.spatial['y']
    t_min, t_max = mechanical_moisture_2d_domain.temporal['t']

    # Visualization grid
    x_vis = np.linspace(x_min, x_max, vis_nx)
    y_vis = np.linspace(y_min, y_max, vis_ny)
    t_vis = np.linspace(t_min, t_max, vis_nt)
    X_vis, Y_vis = np.meshgrid(x_vis, y_vis) # Shape (vis_ny, vis_nx)
    
    # Prepare spatial points for FEM evaluation (must match vis grid for direct comparison)
    evaluation_spatial_points_xy = np.stack((X_vis.ravel(), Y_vis.ravel()), axis=-1)

    # --- Get FEM Ground Truth ---
    if rank == 0:
        print("Running FEM to get ground truth for visualization...")
    
    # Use a reasonable dt for FEM if not specified, could be related to t_vis interval
    fem_dt_actual = fem_dt if fem_dt is not None else (t_max - t_min) / (vis_nt * 5) # Heuristic
    if fem_dt_actual <= 0 : fem_dt_actual = (t_max - t_min) / 100.0


    # The FEM solution will be run by all processes if get_coupled_transient_fem_solution_dolfinx is MPI parallel
    _fem_sol_obj, fem_eval_data_dict = get_coupled_transient_fem(
        mechanical_moisture_2d_domain,
        num_elements_x=fem_nx,
        num_elements_y=fem_ny,
        dt_value=fem_dt_actual,
        evaluation_times=t_vis, # Evaluate FEM at the visualization time points
        evaluation_spatial_points_xy=evaluation_spatial_points_xy
    )
    
    u_fem_gridded = np.full((vis_nt, vis_ny, vis_nx, 2), np.nan) # dim=2 for u_x, u_y
    theta_fem_gridded = np.full((vis_nt, vis_ny, vis_nx, 1), np.nan)

    if fem_eval_data_dict:
        if 'u' in fem_eval_data_dict and fem_eval_data_dict['u'].size > 0:
            # Expected shape from FEM: (vis_nt, vis_ny*vis_nx, dim)
            u_fem_raw = fem_eval_data_dict['u']
            if u_fem_raw.shape[0] == vis_nt and u_fem_raw.shape[1] == (vis_ny * vis_nx):
                u_fem_gridded = u_fem_raw.reshape(vis_nt, vis_ny, vis_nx, 2)
            elif rank == 0:
                print(f"Warning: FEM 'u' data shape mismatch. Got {u_fem_raw.shape}, expected {(vis_nt, vis_ny*vis_nx, 2)}")
        elif rank == 0:
            print("Warning: FEM 'u' data not found or empty in fem_eval_data_dict.")

        if 'theta' in fem_eval_data_dict and fem_eval_data_dict['theta'].size > 0:
            # Expected shape from FEM: (vis_nt, vis_ny*vis_nx, 1)
            theta_fem_raw = fem_eval_data_dict['theta']
            if theta_fem_raw.shape[0] == vis_nt and theta_fem_raw.shape[1] == (vis_ny*vis_nx):
                theta_fem_gridded = theta_fem_raw.reshape(vis_nt, vis_ny, vis_nx, 1)
            elif rank == 0:
                print(f"Warning: FEM 'theta' data shape mismatch. Got {theta_fem_raw.shape}, expected {(vis_nt, vis_ny*vis_nx, 1)}")
        elif rank == 0:
            print("Warning: FEM 'theta' data not found or empty in fem_eval_data_dict.")
    elif rank == 0:
        print("Warning: fem_eval_data_dict is empty.")


    # --- Get PINN Predictions ---
    if rank == 0:
        print("Getting PINN predictions...")
    u_pinn_gridded = np.zeros((vis_nt, vis_ny, vis_nx, 2))
    theta_pinn_gridded = np.zeros((vis_nt, vis_ny, vis_nx, 1))

    for i, t_current in enumerate(t_vis):
        if rank == 0 and i % (vis_nt // 5 + 1) == 0:
            print(f"  PINN predicting for time step {i+1}/{vis_nt}")
        
        T_current_grid = np.full_like(X_vis, t_current) # Shape (vis_ny, vis_nx)
        # Create input for PINN: (N, 3) where N = vis_ny * vis_nx
        pinn_input_points = np.stack((X_vis.ravel(), Y_vis.ravel(), T_current_grid.ravel()), axis=-1)
        pinn_input_scaled = pinn_input_points / np.array([scale.L, scale.L, scale.t])
        
        # PINN model prediction (usually only on rank 0 or if model is broadcasted)
        # Assuming pinn_model.predict can be called by any rank or is handled internally by DeepXDE
        predictions_scaled = pinn_model.predict(pinn_input_scaled) # Expected (N, 3) for u,v,theta
        
        # Unscale and reshape
        u_pinn_gridded[i, :, :, 0] = (predictions_scaled[:, 0] * scale.epsilon).reshape(vis_ny, vis_nx) # u_x
        u_pinn_gridded[i, :, :, 1] = (predictions_scaled[:, 1] * scale.epsilon).reshape(vis_ny, vis_nx) # u_y
        theta_pinn_gridded[i, :, :, 0] = (predictions_scaled[:, 2] * scale.theta).reshape(vis_ny, vis_nx) # theta

    # --- Calculate Differences ---
    # Broadcasting will handle the (..., 1) shape for theta against (...,) if needed
    u_diff_gridded = u_pinn_gridded - u_fem_gridded
    theta_diff_gridded = theta_pinn_gridded - theta_fem_gridded
    
    # --- Animation Setup ---
    if rank == 0: # Matplotlib plotting should ideally be on rank 0
        fig, axes = plt.subplots(3, 3, figsize=(18, 15), sharex=True, sharey=True)
        fig.suptitle(f'PINN vs FEM Comparison (Mechanical-Moisture)', fontsize=16, y=0.97)

        plot_titles = [
            ["PINN u_x", "FEM u_x", "Difference u_x"],
            ["PINN u_y", "FEM u_y", "Difference u_y"],
            ["PINN θ", "FEM θ", "Difference θ"]
        ]
        
        data_to_plot = [
            [u_pinn_gridded[:,:,:,0], u_fem_gridded[:,:,:,0], u_diff_gridded[:,:,:,0]], # u_x data
            [u_pinn_gridded[:,:,:,1], u_fem_gridded[:,:,:,1], u_diff_gridded[:,:,:,1]], # u_y data
            [theta_pinn_gridded[:,:,:,0], theta_fem_gridded[:,:,:,0], theta_diff_gridded[:,:,:,0]] # theta data
        ]

        cmaps_fields = [cm.viridis, cm.viridis, cm.RdBu_r] # PINN, FEM, Diff
        
        ims = [[None]*3 for _ in range(3)]
        cbars = [[None]*3 for _ in range(3)]

        # Determine global vmin/vmax for each row (u_x, u_y, theta) for PINN/FEM, and for Diff
        vmins_maxs = []
        for r in range(3): # u_x, u_y, theta
            pinn_data_row = data_to_plot[r][0]
            fem_data_row = data_to_plot[r][1]
            diff_data_row = data_to_plot[r][2]
            
            # Valid min/max (ignoring NaNs from FEM if any)
            valid_pinn_min = np.nanmin(pinn_data_row)
            valid_pinn_max = np.nanmax(pinn_data_row)
            valid_fem_min = np.nanmin(fem_data_row)
            valid_fem_max = np.nanmax(fem_data_row)
            
            field_min = min(valid_pinn_min, valid_fem_min)
            field_max = max(valid_pinn_max, valid_fem_max)
            
            diff_abs_max = np.nanmax(np.abs(diff_data_row))
            diff_min, diff_max = -diff_abs_max, diff_abs_max
            
            if field_min == field_max: field_max += 1e-9 # Avoid vmin=vmax
            if diff_min == diff_max: diff_max += 1e-9

            vmins_maxs.append([(field_min, field_max), (field_min, field_max), (diff_min, diff_max)])


        for r in range(3): # Rows: u_x, u_y, theta
            for c in range(3): # Cols: PINN, FEM, Difference
                ax = axes[r, c]
                current_data_all_t = data_to_plot[r][c]
                vmin, vmax = vmins_maxs[r][c]

                im = ax.imshow(current_data_all_t[0], origin='lower', aspect='auto',
                               extent=[x_min, x_max, y_min, y_max],
                               cmap=cmaps_fields[c], vmin=vmin, vmax=vmax)
                ims[r][c] = im
                cbars[r][c] = fig.colorbar(im, ax=ax, shrink=0.8)
                ax.set_title(plot_titles[r][c])
                if r == 2: ax.set_xlabel("x (m)")
                if c == 0: ax.set_ylabel("y (m)")

        time_text = fig.text(0.5, 0.93, '', ha='center', fontsize=12)

        def update(frame):
            current_t_val = t_vis[frame]
            time_text.set_text(f'Time = {current_t_val:.2e} s')
            for r_idx in range(3):
                for c_idx in range(3):
                    data_at_frame = data_to_plot[r_idx][c_idx][frame]
                    ims[r_idx][c_idx].set_data(data_at_frame)
            return [im for row_im in ims for im in row_im] + [time_text]

        ani = animation.FuncAnimation(fig, update, frames=vis_nt, interval=interval, blit=True, repeat_delay=1000)
        plt.tight_layout(rect=[0, 0, 1, 0.92]) # Adjust for suptitle and time_text
        
        return {'animation': ani, 'figure': fig}
    else: # Other ranks
        return None # Only rank 0 creates and returns the plot
