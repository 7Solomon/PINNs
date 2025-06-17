import math
from scipy.interpolate import griddata
from utils.fem import evaluate_fem_at_points
from utils.COMSOL import load_comsol_data_mechanic_2d
from utils.metadata import Domain
from process.mechanic.scale import Scale
import numpy as np
import matplotlib.pyplot as plt

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
    GROUND = get_einspannung_2d_fem(fest_lost_2d_domain)
    ground_values_at_points = evaluate_fem_at_points(GROUND, points)
    #print(GROUND)

    # --- Extract min/max for bounds (Needed for outline/limits) ---
    x_min, x_max = fest_lost_2d_domain.spatial['x']
    y_min, y_max = fest_lost_2d_domain.spatial['y']
    # -------------------------------------------------------------

    # Create visualization (3x2 grid)
    fig, axes = plt.subplots(2, 3, figsize=(15, 12))
    fig.suptitle('2D Field Visualization', fontsize=16) # Add an overall title

    # Calculate displacement magnitude (assuming X, Y are now 2D: [ny, nx])
    displacement_magnitude = np.sqrt(predictions[:, 0]**2 + predictions[:, 1]**2)
    displacement_magnitude_2d = displacement_magnitude.reshape(ny, nx)
    
    # Calculate base magnitude for comparison
    #data = load_comsol_data_mechanic_2d('BASELINE/mechanic/einspannung_2d.txt')
    #comsol_u_interp = griddata((data[:,0], data[:,1]), data[:,2], (X.flatten(), Y.flatten()), 
    #                           method='linear', fill_value=0.0)
    #comsol_v_interp = griddata((data[:,0], data[:,1]), data[:,3], (X.flatten(), Y.flatten()), 
    #                           method='linear', fill_value=0.0)
#
    # Calculate base magnitude from interpolated data
    base_magnitude = np.sqrt(ground_values_at_points[:,0]**2 + ground_values_at_points[:,1]**2)
    base_magnitude_2d = base_magnitude.reshape(ny, nx)
    
    # Calculate error
    error_magnitude = np.abs(displacement_magnitude - base_magnitude)
    error_magnitude_2d = error_magnitude.reshape(ny, nx)

    # 1. Displacement Magnitude (Top-Left)
    ax = axes[0, 0]
    contour = ax.contourf(X, Y, displacement_magnitude_2d, levels=20, cmap='viridis')
    ax.set_title("Displacement Magnitude")
    plt.colorbar(contour, ax=ax, shrink=0.8) # Add colorbar
    ax.set_aspect('equal')

    # 2. Base Displacement Magnitude (Top-Center)
    ax = axes[0, 1]
    contour_base = ax.contourf(X, Y, base_magnitude_2d, levels=20, cmap='viridis')
    ax.set_title("Base Displacement Magnitude")
    plt.colorbar(contour_base, ax=ax, shrink=0.8) # Add colorbar
    ax.set_aspect('equal') 

    # 3. Error Magnitude (Top-Right)
    ax = axes[0, 2]
    contour_error = ax.contourf(X, Y, error_magnitude_2d, levels=20, cmap='plasma')
    ax.set_title("Error Magnitude")
    plt.colorbar(contour_error, ax=ax, shrink=0.8) # Add colorbar
    ax.set_aspect('equal')


    # Calculate deformed shape (assuming X, Y are 2D: [ny, nx])
    u_x = predictions[:, 0].reshape(ny, nx)
    u_y = predictions[:, 1].reshape(ny, nx)
    scale_factor = 2.0 
    deformed_X = X - scale_factor * u_x
    deformed_Y = Y - scale_factor * u_y

    # 2. Deformed Shape (Top-Right)
    ax = axes[1, 0]
    ax.scatter(X, Y, c='blue', s=0.5, alpha=0.3, label='Original')
    ax.scatter(deformed_X, deformed_Y, c='red', s=0.5, alpha=0.7, label=f'Deformed (×{scale_factor})')
    ax.set_title("Deformed Shape")
    #ax.legend()
    ax.set_aspect('equal') # Crucial for deformation plots

    # 3. X-Displacement (Bottom-Left)
    ax = axes[1, 1]
    # Create symmetric color limits centered at zero
    u_x_max = np.max(np.abs(predictions[:, 0]))
    scatter_x = ax.scatter(X, Y, c=predictions[:, 0], cmap='RdBu_r', s=1, alpha=0.7, 
                          vmin=-u_x_max, vmax=u_x_max)
    ax.set_title("X-Displacement ($u_x$)")
    plt.colorbar(scatter_x, ax=ax, shrink=0.8)
    ax.set_aspect('equal')

    # 4. Y-Displacement (Bottom-Right)
    ax = axes[1, 2]
    # Create symmetric color limits centered at zero
    u_y_max = np.max(np.abs(predictions[:, 1]))
    scatter_y = ax.scatter(X, Y, c=predictions[:, 1], cmap='RdBu_r', s=1, alpha=0.7,
                          vmin=-u_y_max, vmax=u_y_max)
    ax.set_title("Y-Displacement ($u_y$)")
    plt.colorbar(scatter_y, ax=ax, shrink=0.8)
    ax.set_aspect('equal')
    
    # --- Add beam outline and format all plots ---
    beam_outline = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]])
    
    for ax in axes.flat:
        ax.plot(beam_outline[:, 0], beam_outline[:, 1], 'k-', linewidth=1.5, alpha=0.9, label='Boundary')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Y-coordinate')
        # Set limits with a small buffer
        ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
        ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    # Add legend only once if using the 'Boundary' label
    #axes[0,1].legend() # Re-call legend on one plot to include 'Boundary' if needed

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    
    # plt.savefig("Field_2d_comprehensive.png", dpi=300, bbox_inches='tight')
    # plt.show()
    
    return {'field': fig}


def visualize_comsol_comparison(comsol_data, predictions, fest_lost_2d_domain, **kwargs):
    """
    Visualizes COMSOL data and compares it with model predictions.

    Args:
        comsol_data (np.ndarray): NumPy array with COMSOL data.
                                  Expected columns: [X, Y, U_x_comsol, U_y_comsol]
        predictions (np.ndarray): NumPy array with model predictions.
                                  Expected columns: [U_x_pred, U_y_pred]
                                  Should correspond to the points in comsol_data.
        fest_lost_2d_domain (Domain): Domain object for outline and limits.
    """
    X_comsol = comsol_data[:, 0]
    Y_comsol = comsol_data[:, 1]
    Ux_comsol = comsol_data[:, 2]
    Uy_comsol = comsol_data[:, 3]

    Ux_pred = predictions[:, 0]
    Uy_pred = predictions[:, 1]

    # Calculate deviations
    Ux_deviation = Ux_pred - Ux_comsol
    Uy_deviation = Uy_pred - Uy_comsol

    # --- Extract min/max for bounds (Needed for outline/limits) ---
    x_min, x_max = fest_lost_2d_domain.spatial['x']
    y_min, y_max = fest_lost_2d_domain.spatial['y']
    # -------------------------------------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(17, 14)) # Increased figure size slightly
    fig.suptitle('COMSOL Data and Deviation from Predictions', fontsize=16)

    # 1. COMSOL X-Displacement (Top-Left)
    ax = axes[0, 0]
    ux_comsol_max_abs = np.max(np.abs(Ux_comsol))
    scatter_ux_comsol = ax.scatter(X_comsol, Y_comsol, c=Ux_comsol, cmap='RdBu_r', s=1, alpha=0.7,
                                   vmin=-ux_comsol_max_abs, vmax=ux_comsol_max_abs)
    ax.set_title("COMSOL X-Displacement ($U_x^{COMSOL}$)")
    plt.colorbar(scatter_ux_comsol, ax=ax, shrink=0.8)
    ax.set_aspect('equal')

    # 2. COMSOL Y-Displacement (Top-Right)
    ax = axes[0, 1]
    uy_comsol_max_abs = np.max(np.abs(Uy_comsol))
    scatter_uy_comsol = ax.scatter(X_comsol, Y_comsol, c=Uy_comsol, cmap='RdBu_r', s=1, alpha=0.7,
                                   vmin=-uy_comsol_max_abs, vmax=uy_comsol_max_abs)
    ax.set_title("COMSOL Y-Displacement ($U_y^{COMSOL}$)")
    plt.colorbar(scatter_uy_comsol, ax=ax, shrink=0.8)
    ax.set_aspect('equal')

    # 3. X-Displacement Deviation (Bottom-Left)
    ax = axes[1, 0]
    ux_dev_max_abs = np.max(np.abs(Ux_deviation))
    scatter_ux_dev = ax.scatter(X_comsol, Y_comsol, c=Ux_deviation, cmap='PRGn', s=1, alpha=0.7,
                                vmin=-ux_dev_max_abs, vmax=ux_dev_max_abs) # Using PRGn for deviation
    ax.set_title("X-Displacement Deviation ($U_x^{pred} - U_x^{COMSOL}$)")
    plt.colorbar(scatter_ux_dev, ax=ax, shrink=0.8)
    ax.set_aspect('equal')

    # 4. Y-Displacement Deviation (Bottom-Right)
    ax = axes[1, 1]
    uy_dev_max_abs = np.max(np.abs(Uy_deviation))
    scatter_uy_dev = ax.scatter(X_comsol, Y_comsol, c=Uy_deviation, cmap='PRGn', s=1, alpha=0.7,
                                vmin=-uy_dev_max_abs, vmax=uy_dev_max_abs) # Using PRGn for deviation
    ax.set_title("Y-Displacement Deviation ($U_y^{pred} - U_y^{COMSOL}$)")
    plt.colorbar(scatter_uy_dev, ax=ax, shrink=0.8)
    ax.set_aspect('equal')

    # --- Add beam outline and format all plots ---
    beam_outline = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]])
    
    for ax_idx, ax_row in enumerate(axes):
        for ax_idy, ax in enumerate(ax_row):
            ax.plot(beam_outline[:, 0], beam_outline[:, 1], 'k-', linewidth=1.5, alpha=0.9, label='Boundary')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_xlabel('X-coordinate')
            ax.set_ylabel('Y-coordinate')
            ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
            ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
            #if ax_idx == 0 and ax_idy == 0: # Add legend only once to avoid duplicates
            #     ax.legend(loc='upper right')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    
    # plt.savefig("comsol_comparison.png", dpi=300, bbox_inches='tight')
    # plt.show()
    
    return {'comsol_comparison_plot': fig}