import math
from scipy.interpolate import griddata
from FEM.output import evaluate_solution_at_points_on_rank_0, initialize_point_evaluation, load_fem_results, save_fem_results
from utils.fem import evaluate_fem_at_points
from utils.COMSOL import load_comsol_data_mechanic_2d
from utils.metadata import Domain
from process.mechanic.scale import Scale
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI

from domain_vars import fest_lost_2d_domain
from process.mechanic.gnd import base_mapping, get_einspannung_2d_fem


def visualize_field_1d(model, **kwargs):
    x = np.linspace(0, 1, 1000)[:, None]
    y = model.predict(x)
    y_analytical = base_mapping[type](x)
    #print("max: ", analytical_solution_FLL(1/2))
    plt.figure()
    plt.plot(x, -y, label='NEGATIVE predicted', color='red')

    plt.plot(x, y_analytical, label="Analytical Solution", linestyle='--')
    
    plt.plot(x, np.zeros_like(x), label='Balken', color='black', linewidth=3)
    
    
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("SOlu")
    plt.legend()
    return {'field': plt.gcf()}
    #plt.savefig("Field_1d.png")
    #plt.show()


def visualize_field_2d(model, scale: Scale, **kwargs):
    # Get domain and points
    x_min, x_max = fest_lost_2d_domain.spatial['x']
    y_min, y_max = fest_lost_2d_domain.spatial['y']
    nx, ny = 100, 50  # Number of points in x and y directions
    x_points = np.linspace(x_min, x_max, nx)
    y_points = np.linspace(y_min, y_max, ny)

    X, Y = np.meshgrid(x_points, y_points)
    points = np.vstack((X.flatten(), Y.flatten())).T
    scaled_points = points / np.array([scale.L, scale.L])  # Assuming scale has L attribute for length

    # Get predictions
    predictions = model.predict(scaled_points)
    predictions = predictions * scale.U


    # GROUND
    
    comm = MPI.COMM_WORLD
    GROUND = get_einspannung_2d_fem(fest_lost_2d_domain)
    #GROUND = load_fem_results("BASELINE/mechanic/2d/ground_truth.npy")
    domain = GROUND.function_space.mesh
    _perform_eval, eval_points_3d, bb_tree = initialize_point_evaluation(
        domain, points, comm
    )
    ground_values_at_points = evaluate_solution_at_points_on_rank_0(
        GROUND, eval_points_3d, bb_tree, domain, comm
    )
    save_fem_results("BASELINE/mechanic/2d/ground_truth.npy", ground_values_at_points)


    if comm.rank != 0:
        return {'field': None}

    if ground_values_at_points is None:
        print("Warning: FEM ground truth evaluation failed. Plotting with zeros.")
        ground_values_at_points = np.zeros((points.shape[0], domain.geometry.dim))


    # --- Extract min/max for bounds ---
    x_min, x_max = fest_lost_2d_domain.spatial['x']
    y_min, y_max = fest_lost_2d_domain.spatial['y']
    # -------------------------------------------------------------

    # Create visualization (4x3 grid for detailed comparison)
    fig, axes = plt.subplots(4, 3, figsize=(18, 22)) # Increased size for clarity
    fig.suptitle('2D Field Visualization: Prediction vs. Ground Truth', fontsize=20)

    # --- Data Preparation ---
    # Predicted values
    pred_u_x = predictions[:, 0]
    pred_u_y = predictions[:, 1]
    pred_mag = np.sqrt(pred_u_x**2 + pred_u_y**2)

    # Ground truth values
    gt_u_x = ground_values_at_points[:, 0]
    gt_u_y = ground_values_at_points[:, 1]
    gt_mag = np.sqrt(gt_u_x**2 + gt_u_y**2)

    # Error values
    error_mag = np.abs(pred_mag - gt_mag)
    error_u_x = pred_u_x - gt_u_x
    error_u_y = pred_u_y - gt_u_y

    # Reshape for contour plots
    pred_mag_2d = pred_mag.reshape(ny, nx)
    gt_mag_2d = gt_mag.reshape(ny, nx)
    error_mag_2d = error_mag.reshape(ny, nx)
    
    pred_u_x_2d = pred_u_x.reshape(ny, nx)
    gt_u_x_2d = gt_u_x.reshape(ny, nx)
    error_u_x_2d = error_u_x.reshape(ny, nx)

    pred_u_y_2d = pred_u_y.reshape(ny, nx)
    gt_u_y_2d = gt_u_y.reshape(ny, nx)
    error_u_y_2d = error_u_y.reshape(ny, nx)

    # --- Plotting ---

    # ROW 0: Displacement Magnitude
    ax = axes[0, 0]
    contour = ax.contourf(X, Y, pred_mag_2d, levels=20, cmap='viridis')
    ax.set_title("Predicted Displacement Magnitude")
    plt.colorbar(contour, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    contour_base = ax.contourf(X, Y, gt_mag_2d, levels=20, cmap='viridis')
    ax.set_title("Ground Truth Displacement Magnitude")
    plt.colorbar(contour_base, ax=ax, shrink=0.8)

    ax = axes[0, 2]
    contour_error = ax.contourf(X, Y, error_mag_2d, levels=20, cmap='plasma')
    ax.set_title("Magnitude Error")
    plt.colorbar(contour_error, ax=ax, shrink=0.8)

    # ROW 1: X-Displacement
    ax = axes[1, 0]
    vmax_ux = max(np.abs(pred_u_x).max(), np.abs(gt_u_x).max())
    contour_ux = ax.contourf(X, Y, pred_u_x_2d, levels=20, cmap='RdBu_r')
    ax.set_title("Predicted X-Displacement ($u_x$)")
    plt.colorbar(contour_ux, ax=ax, shrink=0.8)

    ax = axes[1, 1]
    contour_gt_ux = ax.contourf(X, Y, gt_u_x_2d, levels=20, cmap='RdBu_r')
    ax.set_title("Ground Truth X-Displacement ($u_x$)")
    plt.colorbar(contour_gt_ux, ax=ax, shrink=0.8)

    ax = axes[1, 2]
    vmax_err_ux = np.abs(error_u_x).max()
    contour_err_ux = ax.contourf(X, Y, error_u_x_2d, levels=20, cmap='PRGn')
    ax.set_title("X-Displacement Error")
    plt.colorbar(contour_err_ux, ax=ax, shrink=0.8)

    # ROW 2: Y-Displacement
    ax = axes[2, 0]
    vmax_uy = max(np.abs(pred_u_y).max(), np.abs(gt_u_y).max())
    contour_uy = ax.contourf(X, Y, pred_u_y_2d, levels=20, cmap='RdBu_r')
    ax.set_title("Predicted Y-Displacement ($u_y$)")
    plt.colorbar(contour_uy, ax=ax, shrink=0.8)

    ax = axes[2, 1]
    contour_gt_uy = ax.contourf(X, Y, gt_u_y_2d, levels=20, cmap='RdBu_r')
    ax.set_title("Ground Truth Y-Displacement ($u_y$)")
    plt.colorbar(contour_gt_uy, ax=ax, shrink=0.8)

    ax = axes[2, 2]
    vmax_err_uy = np.abs(error_u_y).max()
    contour_err_uy = ax.contourf(X, Y, error_u_y_2d, levels=20, cmap='PRGn')
    ax.set_title("Y-Displacement Error")
    plt.colorbar(contour_err_uy, ax=ax, shrink=0.8)

    # ROW 3: Deformed Shape
    scale_factor = 10.0
    
    # Predicted Deformed Shape
    ax = axes[3, 0]
    deformed_X_pred = X + scale_factor * pred_u_x_2d
    deformed_Y_pred = Y + scale_factor * pred_u_y_2d
    ax.scatter(X, Y, c='blue', s=0.5, alpha=0.3, label='Original')
    ax.scatter(deformed_X_pred, deformed_Y_pred, c='red', s=0.5, alpha=0.7, label=f'Deformed (×{scale_factor})')
    ax.set_title("Predicted Deformed Shape")
    ax.legend()

    # Ground Truth Deformed Shape
    ax = axes[3, 1]
    deformed_X_gt = X + scale_factor * gt_u_x_2d
    deformed_Y_gt = Y + scale_factor * gt_u_y_2d
    ax.scatter(X, Y, c='blue', s=0.5, alpha=0.3, label='Original')
    ax.scatter(deformed_X_gt, deformed_Y_gt, c='green', s=0.5, alpha=0.7, label=f'Deformed (×{scale_factor})')
    ax.set_title("Ground Truth Deformed Shape")
    ax.legend()

    # Turn off the last unused plot
    axes[3, 2].axis('off')

    # --- Add beam outline and format all plots ---
    beam_outline = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]])
    
    for ax in axes.flat:
        # Skip formatting for the turned-off axis which has no title
        if not ax.get_title(): 
            continue
        ax.plot(beam_outline[:, 0], beam_outline[:, 1], 'k-', linewidth=1.5, alpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Y-coordinate')
        ax.set_aspect('equal')
        # Set limits with a small buffer
        ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
        ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to make space for suptitle
    
    return {'field': fig}


#def visualize_comsol_comparison(comsol_data, predictions, fest_lost_2d_domain, **kwargs):
#    """
#    Visualizes COMSOL data and compares it with model predictions.
#
#    Args:
#        comsol_data (np.ndarray): NumPy array with COMSOL data.
#                                  Expected columns: [X, Y, U_x_comsol, U_y_comsol]
#        predictions (np.ndarray): NumPy array with model predictions.
#                                  Expected columns: [U_x_pred, U_y_pred]
#                                  Should correspond to the points in comsol_data.
#        fest_lost_2d_domain (Domain): Domain object for outline and limits.
#    """
#    X_comsol = comsol_data[:, 0]
#    Y_comsol = comsol_data[:, 1]
#    Ux_comsol = comsol_data[:, 2]
#    Uy_comsol = comsol_data[:, 3]
#
#    Ux_pred = predictions[:, 0]
#    Uy_pred = predictions[:, 1]
#
#    # Calculate deviations
#    Ux_deviation = Ux_pred - Ux_comsol
#    Uy_deviation = Uy_pred - Uy_comsol
#
#    # --- Extract min/max for bounds (Needed for outline/limits) ---
#    x_min, x_max = fest_lost_2d_domain.spatial['x']
#    y_min, y_max = fest_lost_2d_domain.spatial['y']
#    # -------------------------------------------------------------
#
#    fig, axes = plt.subplots(2, 2, figsize=(17, 14)) # Increased figure size slightly
#    fig.suptitle('COMSOL Data and Deviation from Predictions', fontsize=16)
#
#    # 1. COMSOL X-Displacement (Top-Left)
#    ax = axes[0, 0]
#    ux_comsol_max_abs = np.max(np.abs(Ux_comsol))
#    scatter_ux_comsol = ax.scatter(X_comsol, Y_comsol, c=Ux_comsol, cmap='RdBu_r', s=1, alpha=0.7,
#                                   vmin=-ux_comsol_max_abs, vmax=ux_comsol_max_abs)
#    ax.set_title("COMSOL X-Displacement ($U_x^{COMSOL}$)")
#    plt.colorbar(scatter_ux_comsol, ax=ax, shrink=0.8)
#    ax.set_aspect('equal')
#
#    # 2. COMSOL Y-Displacement (Top-Right)
#    ax = axes[0, 1]
#    uy_comsol_max_abs = np.max(np.abs(Uy_comsol))
#    scatter_uy_comsol = ax.scatter(X_comsol, Y_comsol, c=Uy_comsol, cmap='RdBu_r', s=1, alpha=0.7,
#                                   vmin=-uy_comsol_max_abs, vmax=uy_comsol_max_abs)
#    ax.set_title("COMSOL Y-Displacement ($U_y^{COMSOL}$)")
#    plt.colorbar(scatter_uy_comsol, ax=ax, shrink=0.8)
#    ax.set_aspect('equal')
#
#    # 3. X-Displacement Deviation (Bottom-Left)
#    ax = axes[1, 0]
#    ux_dev_max_abs = np.max(np.abs(Ux_deviation))
#    scatter_ux_dev = ax.scatter(X_comsol, Y_comsol, c=Ux_deviation, cmap='PRGn', s=1, alpha=0.7,
#                                vmin=-ux_dev_max_abs, vmax=ux_dev_max_abs) # Using PRGn for deviation
#    ax.set_title("X-Displacement Deviation ($U_x^{pred} - U_x^{COMSOL}$)")
#    plt.colorbar(scatter_ux_dev, ax=ax, shrink=0.8)
#    ax.set_aspect('equal')
#
#    # 4. Y-Displacement Deviation (Bottom-Right)
#    ax = axes[1, 1]
#    uy_dev_max_abs = np.max(np.abs(Uy_deviation))
#    scatter_uy_dev = ax.scatter(X_comsol, Y_comsol, c=Uy_deviation, cmap='PRGn', s=1, alpha=0.7,
#                                vmin=-uy_dev_max_abs, vmax=uy_dev_max_abs) # Using PRGn for deviation
#    ax.set_title("Y-Displacement Deviation ($U_y^{pred} - U_y^{COMSOL}$)")
#    plt.colorbar(scatter_uy_dev, ax=ax, shrink=0.8)
#    ax.set_aspect('equal')
#
#    # --- Add beam outline and format all plots ---
#    beam_outline = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]])
#    
#    for ax_idx, ax_row in enumerate(axes):
#        for ax_idy, ax in enumerate(ax_row):
#            ax.plot(beam_outline[:, 0], beam_outline[:, 1], 'k-', linewidth=1.5, alpha=0.9, label='Boundary')
#            ax.grid(True, linestyle='--', alpha=0.4)
#            ax.set_xlabel('X-coordinate')
#            ax.set_ylabel('Y-coordinate')
#            ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
#            ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
#            #if ax_idx == 0 and ax_idy == 0: # Add legend only once to avoid duplicates
#            #     ax.legend(loc='upper right')
#
#
#    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
#    
#    # plt.savefig("comsol_comparison.png", dpi=300, bbox_inches='tight')
#    # plt.show()
#    
#    return {'comsol_comparison_plot': fig}