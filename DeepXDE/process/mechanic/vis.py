import math
from scipy.interpolate import griddata
from vis import get_deformation_amplifier
from FEM.output import evaluate_solution_at_points_on_rank_0, initialize_point_evaluation, load_fem_results, save_fem_results
from utils.fem import evaluate_fem_at_points
from utils.COMSOL import load_comsol_data_mechanic_2d
from utils.metadata import Domain
from process.mechanic.scale import EnsemnbleMechanicScale, MechanicScale
import numpy as np
import matplotlib.pyplot as plt


from mpi4py import MPI
import dolfinx as df

from domain_vars import  einspannung_2d_domain
from process.mechanic.gnd import base_mapping, get_einspannung_2d_fem, get_ensemble_einspannung_2d_fem_points, get_sigma_fem


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


def visualize_field_2d(model, scale: MechanicScale, points_data: dict, fem_value_points: np.ndarray, **kwargs):
    # Get domain and points using the new, structured points_data
    x_min, x_max = einspannung_2d_domain.spatial['x']
    y_min, y_max = einspannung_2d_domain.spatial['y']
    
    nx, ny = points_data['resolution']['x'], points_data['resolution']['y']
    
    X = points_data['spatial_meshgrid']['ij']['x']
    Y = points_data['spatial_meshgrid']['ij']['y']
    
    # The rest of your scaling and prediction logic is fine
    scaled_points = points_data['spatial_points_flat'] / np.array([scale.L, scale.L])
    predictions = model.predict(scaled_points)
    predictions = points_data['reshape_utils']['pred_to_ij'](predictions)
    predictions = predictions * scale.value_scale_list
    print('fem_value_points', fem_value_points.shape, fem_value_points.min().item(), fem_value_points.max().item())

    fields = visualize_field_2d_test(predictions, fem_value_points, X, Y, [(x_min, x_max), (y_min, y_max)])
    return fields


def visualize_field_2d_test(predictions, fem_value_points, X, Y, dim):
    (x_min, x_max), (y_min, y_max) = dim
    
    # Calculate all the data arrays
    pred_u_x = predictions[:, :, 0]
    pred_u_y = predictions[:, :, 1]
    pred_mag = np.sqrt(pred_u_x**2 + pred_u_y**2)

    gt_u_x = fem_value_points[:, :, 0]
    gt_u_y = fem_value_points[:, :, 1]
    gt_mag = np.sqrt(gt_u_x**2 + gt_u_y**2)

    # Error values
    error_mag_2d = np.abs(pred_mag - gt_mag)
    error_u_x_2d = pred_u_x - gt_u_x
    error_u_y_2d = pred_u_y - gt_u_y

    aspect_ratio = (x_max - x_min) / (y_max - y_min) * 0.1
    beam_outline = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]])
    
    # Helper function to format individual plots
    def format_axis(ax, title):
        ax.plot(beam_outline[:, 0], beam_outline[:, 1], 'k-', linewidth=1.5, alpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Y-coordinate')
        ax.set_aspect(aspect_ratio)
        ax.set_title(title)
        ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
        ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    # Dictionary to store individual plots
    fields = {}

    # Predicted Displacement Magnitude
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, pred_mag, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, "Predicted Displacement Magnitude")
    fields['pred_mag'] = fig

    # Ground Truth Displacement Magnitude
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, gt_mag, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, "Ground Truth Displacement Magnitude")
    fields['ground_mag'] = fig

    # Magnitude Error
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, error_mag_2d, levels=20, cmap='plasma')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, "Magnitude Error")
    fields['error_mag'] = fig

    # Predicted X-Displacement
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, pred_u_x, levels=20, cmap='RdBu_r')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, "Predicted X-Displacement ($u_x$)")
    fields['pred_u_x'] = fig

    # Ground Truth X-Displacement
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, gt_u_x, levels=20, cmap='RdBu_r')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, "Ground Truth X-Displacement ($u_x$)")
    fields['ground_u_x'] = fig

    # X-Displacement Error
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, error_u_x_2d, levels=20, cmap='PRGn')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, "X-Displacement Error")
    fields['error_u_x'] = fig

    # Predicted Y-Displacement
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, pred_u_y, levels=20, cmap='RdBu_r')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, "Predicted Y-Displacement ($u_y$)")
    fields['pred_u_y'] = fig

    # Ground Truth Y-Displacement
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, gt_u_y, levels=20, cmap='RdBu_r')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, "Ground Truth Y-Displacement ($u_y$)")
    fields['ground_u_y'] = fig

    # Y-Displacement Error
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, error_u_y_2d, levels=20, cmap='PRGn')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, "Y-Displacement Error")
    fields['error_u_y'] = fig

    # Predicted Deformed Shape
    scale_factor = 10
    fig, ax = plt.subplots(figsize=(8, 6))
    deformed_X_pred = X + scale_factor * pred_u_x
    deformed_Y_pred = Y + scale_factor * pred_u_y
    ax.scatter(X, Y, c='blue', s=0.5, alpha=0.3, label='Original')
    ax.scatter(deformed_X_pred, deformed_Y_pred, c='red', s=0.5, alpha=0.7, label=f'Deformed (×{scale_factor})')
    format_axis(ax, f"Predicted Deformed Shape (×{scale_factor})")
    fields['pred_deformation'] = fig

    # Ground Truth Deformed Shape
    fig, ax = plt.subplots(figsize=(8, 6))
    deformed_X_gt = X + scale_factor * gt_u_x
    deformed_Y_gt = Y + scale_factor * gt_u_y
    ax.scatter(X, Y, c='blue', s=0.5, alpha=0.3, label='Original')
    ax.scatter(deformed_X_gt, deformed_Y_gt, c='green', s=0.5, alpha=0.7, label=f'Deformed (×{scale_factor})')
    format_axis(ax, f"Ground Truth Deformed Shape (×{scale_factor})")
    fields['ground_deformation'] = fig

    # Create the combined overview plot
    fig_combined, axes = plt.subplots(4, 3, figsize=(18, 22))
    fig_combined.suptitle('2D Field Visualization: Prediction vs. Ground Truth', fontsize=20)
    
    # ROW 0: Displacement Magnitude
    ax = axes[0, 0]
    contour = ax.contourf(X, Y, pred_mag, levels=20, cmap='viridis')
    ax.set_title("Predicted Displacement Magnitude")
    plt.colorbar(contour, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    contour_base = ax.contourf(X, Y, gt_mag, levels=20, cmap='viridis')
    ax.set_title("Ground Truth Displacement Magnitude")
    plt.colorbar(contour_base, ax=ax, shrink=0.8)

    ax = axes[0, 2]
    contour_error = ax.contourf(X, Y, error_mag_2d, levels=20, cmap='plasma')
    ax.set_title("Magnitude Error")
    plt.colorbar(contour_error, ax=ax, shrink=0.8)

    # ROW 1: X-Displacement
    ax = axes[1, 0]
    contour_ux = ax.contourf(X, Y, pred_u_x, levels=20, cmap='RdBu_r')
    ax.set_title("Predicted X-Displacement ($u_x$)")
    plt.colorbar(contour_ux, ax=ax, shrink=0.8)

    ax = axes[1, 1]
    contour_gt_ux = ax.contourf(X, Y, gt_u_x, levels=20, cmap='RdBu_r')
    ax.set_title("Ground Truth X-Displacement ($u_x$)")
    plt.colorbar(contour_gt_ux, ax=ax, shrink=0.8)

    ax = axes[1, 2]
    contour_err_ux = ax.contourf(X, Y, error_u_x_2d, levels=20, cmap='PRGn')
    ax.set_title("X-Displacement Error")
    plt.colorbar(contour_err_ux, ax=ax, shrink=0.8)

    # ROW 2: Y-Displacement
    ax = axes[2, 0]
    contour_uy = ax.contourf(X, Y, pred_u_y, levels=20, cmap='RdBu_r')
    ax.set_title("Predicted Y-Displacement ($u_y$)")
    plt.colorbar(contour_uy, ax=ax, shrink=0.8)

    ax = axes[2, 1]
    contour_gt_uy = ax.contourf(X, Y, gt_u_y, levels=20, cmap='RdBu_r')
    ax.set_title("Ground Truth Y-Displacement ($u_y$)")
    plt.colorbar(contour_gt_uy, ax=ax, shrink=0.8)

    ax = axes[2, 2]
    contour_err_uy = ax.contourf(X, Y, error_u_y_2d, levels=20, cmap='PRGn')
    ax.set_title("Y-Displacement Error")
    plt.colorbar(contour_err_uy, ax=ax, shrink=0.8)

    # ROW 3: Deformed Shapes
    scale_factor = 10
    
    # Predicted Deformed Shape
    ax = axes[3, 0]
    deformed_X_pred = X + scale_factor * pred_u_x
    deformed_Y_pred = Y + scale_factor * pred_u_y
    ax.scatter(X, Y, c='blue', s=0.5, alpha=0.3, label='Original')
    ax.scatter(deformed_X_pred, deformed_Y_pred, c='red', s=0.5, alpha=0.7, label=f'Deformed (×{scale_factor})')
    ax.set_title(f"Predicted Deformed Shape (×{scale_factor})")

    # Ground Truth Deformed Shape
    ax = axes[3, 1]
    deformed_X_gt = X + scale_factor * gt_u_x
    deformed_Y_gt = Y + scale_factor * gt_u_y
    ax.scatter(X, Y, c='blue', s=0.5, alpha=0.3, label='Original')
    ax.scatter(deformed_X_gt, deformed_Y_gt, c='green', s=0.5, alpha=0.7, label=f'Deformed (×{scale_factor})')
    ax.set_title(f"Ground Truth Deformed Shape (×{scale_factor})")

    # Turn off the last unused plot
    axes[3, 2].axis('off')

    # --- Add beam outline and format all plots ---
    beam_outline = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]])
    
    for ax in axes.flat:
        # Skip formatting for the turned-off axis
        if not ax.get_title(): 
            continue
        ax.plot(beam_outline[:, 0], beam_outline[:, 1], 'k-', linewidth=1.5, alpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Y-coordinate')
        ax.set_aspect(aspect_ratio)
        ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
        ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    fields['combined_overview'] = fig_combined

    return fields

    

def visualize_field_2d_ensemble(model, scale: EnsemnbleMechanicScale, points_data: dict, fem_value_points: np.ndarray, **kwargs):
    # Get domain and points using the new, structured points_data
    x_min, x_max = einspannung_2d_domain.spatial['x']
    y_min, y_max = einspannung_2d_domain.spatial['y']
    
    nx, ny = points_data['resolution']['x'], points_data['resolution']['y']
    
    X = points_data['spatial_meshgrid']['ij']['x']
    Y = points_data['spatial_meshgrid']['ij']['y']
    
    # The rest of your scaling and prediction logic is fine
    scaled_points = points_data['spatial_points_flat'] / np.array([scale.L, scale.L])
    predictions = model.predict(scaled_points)
    predictions = points_data['reshape_utils']['pred_to_ij'](predictions)
    predictions = predictions * scale.value_scale_list
    fields = visualize_field_2d_ensemble_test(predictions, fem_value_points, X, Y, [(x_min, x_max), (y_min, y_max)])
    return fields

def visualize_field_2d_ensemble_test(predictions, fem_value_points, X, Y, dim):
    (x_min, x_max), (y_min, y_max) = dim
    
    # Updated for V2 residual: [ux, uy, exx, eyy, exy, σxx, σyy, τxy]
    pred_u_x = predictions[:, :, 0]
    pred_u_y = predictions[:, :, 1]
    pred_e_xx = predictions[:, :, 2]  # strain components
    pred_e_yy = predictions[:, :, 3]
    pred_e_xy = predictions[:, :, 4]
    pred_sigma_xx = predictions[:, :, 5]  # stress components now at indices 5,6,7
    pred_sigma_yy = predictions[:, :, 6]
    pred_tau_xy = predictions[:, :, 7]
    pred_mag = np.sqrt(pred_u_x**2 + pred_u_y**2)

    # Ground truth - assuming same structure
    gt_u_x = fem_value_points[:, :, 0]
    gt_u_y = fem_value_points[:, :, 1]
    gt_e_xx = fem_value_points[:, :, 2]
    gt_e_yy = fem_value_points[:, :, 3]
    gt_e_xy = fem_value_points[:, :, 4]
    gt_sigma_xx = fem_value_points[:, :, 5]
    gt_sigma_yy = fem_value_points[:, :, 6]
    gt_tau_xy = fem_value_points[:, :, 7]
    gt_mag = np.sqrt(gt_u_x**2 + gt_u_y**2)

    # Error values for all components
    error_mag_2d = np.abs(pred_mag - gt_mag)
    error_u_x_2d = pred_u_x - gt_u_x
    error_u_y_2d = pred_u_y - gt_u_y
    error_e_xx_2d = pred_e_xx - gt_e_xx  # strain errors
    error_e_yy_2d = pred_e_yy - gt_e_yy
    error_e_xy_2d = pred_e_xy - gt_e_xy
    error_sigma_xx_2d = pred_sigma_xx - gt_sigma_xx
    error_sigma_yy_2d = pred_sigma_yy - gt_sigma_yy
    error_tau_xy_2d = pred_tau_xy - gt_tau_xy

    aspect_ratio = (x_max - x_min) / (y_max - y_min) * 0.1
    beam_outline = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]])
    
    # Helper function to format individual plots
    def format_axis(ax, title):
        ax.plot(beam_outline[:, 0], beam_outline[:, 1], 'k-', linewidth=1.5, alpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Y-coordinate')
        ax.set_aspect(aspect_ratio)
        ax.set_title(title)
        ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
        ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    fields = {}

    # === DISPLACEMENT MAGNITUDE PLOTS ===
    # Predicted Displacement Magnitude
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, pred_mag, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, "Predicted Displacement Magnitude")
    fields['pred_mag'] = fig

    # Ground Truth Displacement Magnitude
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, gt_mag, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, "Ground Truth Displacement Magnitude")
    fields['ground_mag'] = fig

    # Magnitude Error
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, error_mag_2d, levels=20, cmap='plasma')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, "Magnitude Error")
    fields['error_mag'] = fig

    # === X-DISPLACEMENT PLOTS ===
    # Predicted X-Displacement
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, pred_u_x, levels=20, cmap='RdBu_r')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, "Predicted X-Displacement ($u_x$)")
    fields['pred_u_x'] = fig

    # Ground Truth X-Displacement
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, gt_u_x, levels=20, cmap='RdBu_r')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, "Ground Truth X-Displacement ($u_x$)")
    fields['ground_u_x'] = fig

    # X-Displacement Error
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, error_u_x_2d, levels=20, cmap='PRGn')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, "X-Displacement Error")
    fields['error_u_x'] = fig

    # === Y-DISPLACEMENT PLOTS ===
    # Predicted Y-Displacement
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, pred_u_y, levels=20, cmap='RdBu_r')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, "Predicted Y-Displacement ($u_y$)")
    fields['pred_u_y'] = fig

    # Ground Truth Y-Displacement
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, gt_u_y, levels=20, cmap='RdBu_r')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, "Ground Truth Y-Displacement ($u_y$)")
    fields['ground_u_y'] = fig

    # Y-Displacement Error
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, error_u_y_2d, levels=20, cmap='PRGn')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, "Y-Displacement Error")
    fields['error_u_y'] = fig

    # === STRAIN EXX PLOTS ===
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, pred_e_xx, levels=20, cmap='coolwarm')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, r"Predicted Strain $\varepsilon_{xx}$")
    fields['pred_e_xx'] = fig

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, gt_e_xx, levels=20, cmap='coolwarm')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, r"Ground Truth Strain $\varepsilon_{xx}$")
    fields['ground_e_xx'] = fig

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, error_e_xx_2d, levels=20, cmap='PRGn')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, r"Strain $\varepsilon_{xx}$ Error")
    fields['error_e_xx'] = fig

    # === STRAIN EYY PLOTS ===
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, pred_e_yy, levels=20, cmap='coolwarm')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, r"Predicted Strain $\varepsilon_{yy}$")
    fields['pred_e_yy'] = fig

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, gt_e_yy, levels=20, cmap='coolwarm')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, r"Ground Truth Strain $\varepsilon_{yy}$")
    fields['ground_e_yy'] = fig

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, error_e_yy_2d, levels=20, cmap='PRGn')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, r"Strain $\varepsilon_{yy}$ Error")
    fields['error_e_yy'] = fig

    # === STRAIN EXY PLOTS ===
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, pred_e_xy, levels=20, cmap='coolwarm')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, r"Predicted Strain $\varepsilon_{xy}$")
    fields['pred_e_xy'] = fig

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, gt_e_xy, levels=20, cmap='coolwarm')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, r"Ground Truth Strain $\varepsilon_{xy}$")
    fields['ground_e_xy'] = fig

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, error_e_xy_2d, levels=20, cmap='PRGn')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, r"Strain $\varepsilon_{xy}$ Error")
    fields['error_e_xy'] = fig

    # === STRESS SIGMA_XX PLOTS ===
    # Predicted Sigma_XX
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, pred_sigma_xx, levels=20, cmap='coolwarm')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, r"Predicted Stress $\sigma_{xx}$")
    fields['pred_sigma_xx'] = fig

    # Ground Truth Sigma_XX
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, gt_sigma_xx, levels=20, cmap='coolwarm')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, r"Ground Truth Stress $\sigma_{xx}$")
    fields['ground_sigma_xx'] = fig

    # Sigma_XX Error
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, error_sigma_xx_2d, levels=20, cmap='PRGn')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, r"Stress $\sigma_{xx}$ Error")
    fields['error_sigma_xx'] = fig

    # === STRESS SIGMA_YY PLOTS ===
    # Predicted Sigma_YY
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, pred_sigma_yy, levels=20, cmap='coolwarm')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, r"Predicted Stress $\sigma_{yy}$")
    fields['pred_sigma_yy'] = fig

    # Ground Truth Sigma_YY
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, gt_sigma_yy, levels=20, cmap='coolwarm')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, r"Ground Truth Stress $\sigma_{yy}$")
    fields['ground_sigma_yy'] = fig

    # Sigma_YY Error
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, error_sigma_yy_2d, levels=20, cmap='PRGn')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, r"Stress $\sigma_{yy}$ Error")
    fields['error_sigma_yy'] = fig

    # === STRESS TAU_XY PLOTS ===
    # Predicted Tau_XY
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, pred_tau_xy, levels=20, cmap='coolwarm')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, r"Predicted Stress $\tau_{xy}$")
    fields['pred_tau_xy'] = fig

    # Ground Truth Tau_XY
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, gt_tau_xy, levels=20, cmap='coolwarm')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, r"Ground Truth Stress $\tau_{xy}$")
    fields['ground_tau_xy'] = fig

    # Tau_XY Error
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, error_tau_xy_2d, levels=20, cmap='PRGn')
    plt.colorbar(contour, ax=ax, shrink=0.8)
    format_axis(ax, r"Stress $\tau_{xy}$ Error")
    fields['error_tau_xy'] = fig

    # === DEFORMED SHAPE PLOTS ===
    scale_factor = 10
    
    # Predicted Deformed Shape
    fig, ax = plt.subplots(figsize=(8, 6))
    deformed_X_pred = X + scale_factor * pred_u_x
    deformed_Y_pred = Y + scale_factor * pred_u_y
    ax.scatter(X, Y, c='blue', s=0.5, alpha=0.3, label='Original')
    ax.scatter(deformed_X_pred, deformed_Y_pred, c='red', s=0.5, alpha=0.7, label=f'Deformed (×{scale_factor})')
    format_axis(ax, f"Predicted Deformed Shape (×{scale_factor})")
    ax.legend()
    fields['pred_deformation'] = fig

    # Ground Truth Deformed Shape
    fig, ax = plt.subplots(figsize=(8, 6))
    deformed_X_gt = X + scale_factor * gt_u_x
    deformed_Y_gt = Y + scale_factor * gt_u_y
    ax.scatter(X, Y, c='blue', s=0.5, alpha=0.3, label='Original')
    ax.scatter(deformed_X_gt, deformed_Y_gt, c='green', s=0.5, alpha=0.7, label=f'Deformed (×{scale_factor})')
    format_axis(ax, f"Ground Truth Deformed Shape (×{scale_factor})")
    ax.legend()
    fields['ground_deformation'] = fig

    # === COMBINED OVERVIEW PLOT ===
    fig_combined, axes = plt.subplots(8, 3, figsize=(18, 30))  # Increased from 6 to 8 rows
    fig_combined.suptitle('2D Field Visualization: Prediction vs. Ground Truth (V2)', fontsize=20)

    # ROW 0: Displacement Magnitude
    ax = axes[0, 0]
    contour = ax.contourf(X, Y, pred_mag, levels=20, cmap='viridis')
    ax.set_title("Predicted Displacement Magnitude")
    plt.colorbar(contour, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    contour_base = ax.contourf(X, Y, gt_mag, levels=20, cmap='viridis')
    ax.set_title("Ground Truth Displacement Magnitude")
    plt.colorbar(contour_base, ax=ax, shrink=0.8)

    ax = axes[0, 2]
    contour_error = ax.contourf(X, Y, error_mag_2d, levels=20, cmap='plasma')
    ax.set_title("Magnitude Error")
    plt.colorbar(contour_error, ax=ax, shrink=0.8)

    # ROW 1: X-Displacement
    ax = axes[1, 0]
    contour_ux = ax.contourf(X, Y, pred_u_x, levels=20, cmap='RdBu_r')
    ax.set_title("Predicted X-Displacement ($u_x$)")
    plt.colorbar(contour_ux, ax=ax, shrink=0.8)

    ax = axes[1, 1]
    contour_gt_ux = ax.contourf(X, Y, gt_u_x, levels=20, cmap='RdBu_r')
    ax.set_title("Ground Truth X-Displacement ($u_x$)")
    plt.colorbar(contour_gt_ux, ax=ax, shrink=0.8)

    ax = axes[1, 2]
    contour_err_ux = ax.contourf(X, Y, error_u_x_2d, levels=20, cmap='PRGn')
    ax.set_title("X-Displacement Error")
    plt.colorbar(contour_err_ux, ax=ax, shrink=0.8)

    # ROW 2: Y-Displacement
    ax = axes[2, 0]
    contour_uy = ax.contourf(X, Y, pred_u_y, levels=20, cmap='RdBu_r')
    ax.set_title("Predicted Y-Displacement ($u_y$)")
    plt.colorbar(contour_uy, ax=ax, shrink=0.8)

    ax = axes[2, 1]
    contour_gt_uy = ax.contourf(X, Y, gt_u_y, levels=20, cmap='RdBu_r')
    ax.set_title("Ground Truth Y-Displacement ($u_y$)")
    plt.colorbar(contour_gt_uy, ax=ax, shrink=0.8)

    ax = axes[2, 2]
    contour_err_uy = ax.contourf(X, Y, error_u_y_2d, levels=20, cmap='PRGn')
    ax.set_title("Y-Displacement Error")
    plt.colorbar(contour_err_uy, ax=ax, shrink=0.8)

    # ROW 3: Strain EXX
    ax = axes[3, 0]
    contour = ax.contourf(X, Y, pred_e_xx, levels=20, cmap='coolwarm')
    ax.set_title(r"Predicted Strain $\varepsilon_{xx}$")
    plt.colorbar(contour, ax=ax, shrink=0.8)

    ax = axes[3, 1]
    contour = ax.contourf(X, Y, gt_e_xx, levels=20, cmap='coolwarm')
    ax.set_title(r"Ground Truth Strain $\varepsilon_{xx}$")
    plt.colorbar(contour, ax=ax, shrink=0.8)

    ax = axes[3, 2]
    contour = ax.contourf(X, Y, error_e_xx_2d, levels=20, cmap='PRGn')
    ax.set_title(r"Strain $\varepsilon_{xx}$ Error")
    plt.colorbar(contour, ax=ax, shrink=0.8)

    # ROW 4: Strain EYY
    ax = axes[4, 0]
    contour = ax.contourf(X, Y, pred_e_yy, levels=20, cmap='coolwarm')
    ax.set_title(r"Predicted Strain $\varepsilon_{yy}$")
    plt.colorbar(contour, ax=ax, shrink=0.8)

    ax = axes[4, 1]
    contour = ax.contourf(X, Y, gt_e_yy, levels=20, cmap='coolwarm')
    ax.set_title(r"Ground Truth Strain $\varepsilon_{yy}$")
    plt.colorbar(contour, ax=ax, shrink=0.8)

    ax = axes[4, 2]
    contour = ax.contourf(X, Y, error_e_yy_2d, levels=20, cmap='PRGn')
    ax.set_title(r"Strain $\varepsilon_{yy}$ Error")
    plt.colorbar(contour, ax=ax, shrink=0.8)

    # ROW 5: Strain EXY
    ax = axes[5, 0]
    contour = ax.contourf(X, Y, pred_e_xy, levels=20, cmap='coolwarm')
    ax.set_title(r"Predicted Strain $\varepsilon_{xy}$")
    plt.colorbar(contour, ax=ax, shrink=0.8)

    ax = axes[5, 1]
    contour = ax.contourf(X, Y, gt_e_xy, levels=20, cmap='coolwarm')
    ax.set_title(r"Ground Truth Strain $\varepsilon_{xy}$")
    plt.colorbar(contour, ax=ax, shrink=0.8)

    ax = axes[5, 2]
    contour = ax.contourf(X, Y, error_e_xy_2d, levels=20, cmap='PRGn')
    ax.set_title(r"Strain $\varepsilon_{xy}$ Error")
    plt.colorbar(contour, ax=ax, shrink=0.8)

    # ROW 6: Stress Components
    ax = axes[6, 0]
    contour_sigma_xx = ax.contourf(X, Y, pred_sigma_xx, levels=20, cmap='coolwarm')
    ax.set_title(r"Predicted Stress $\sigma_{xx}$")
    plt.colorbar(contour_sigma_xx, ax=ax, shrink=0.8)

    ax = axes[6, 1]
    contour_sigma_yy = ax.contourf(X, Y, pred_sigma_yy, levels=20, cmap='coolwarm')
    ax.set_title(r"Predicted Stress $\sigma_{yy}$")
    plt.colorbar(contour_sigma_yy, ax=ax, shrink=0.8)

    ax = axes[6, 2]
    contour_tau_xy = ax.contourf(X, Y, pred_tau_xy, levels=20, cmap='coolwarm')
    ax.set_title(r"Predicted Stress $\tau_{xy}$")
    plt.colorbar(contour_tau_xy, ax=ax, shrink=0.8)

    # ROW 7: Error values for stress components
    ax = axes[7, 0]
    contour_err_sigma_xx = ax.contourf(X, Y, error_sigma_xx_2d, levels=20, cmap='PRGn')
    ax.set_title(r"Stress $\sigma_{xx}$ Error")
    plt.colorbar(contour_err_sigma_xx, ax=ax, shrink=0.8)

    ax = axes[7, 1]
    contour_err_sigma_yy = ax.contourf(X, Y, error_sigma_yy_2d, levels=20, cmap='PRGn')
    ax.set_title(r"Stress $\sigma_{yy}$ Error")
    plt.colorbar(contour_err_sigma_yy, ax=ax, shrink=0.8)

    ax = axes[7, 2]
    contour_err_tau_xy = ax.contourf(X, Y, error_tau_xy_2d, levels=20, cmap='PRGn')
    ax.set_title(r"Stress $\tau_{xy}$ Error")
    plt.colorbar(contour_err_tau_xy, ax=ax, shrink=0.8)

    beam_outline = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]])
    
    for i in range(8):  # 8 rows
        for j in range(3):  # 3 columns
            ax = axes[i, j]
            # Format each subplot
            ax.plot(beam_outline[:, 0], beam_outline[:, 1], 'k-', linewidth=1.5, alpha=0.9)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_xlabel('X-coordinate')
            ax.set_ylabel('Y-coordinate')
            ax.set_aspect(aspect_ratio)
            ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
            ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    fields['combined_overview'] = fig_combined

    return fields
    return fields

