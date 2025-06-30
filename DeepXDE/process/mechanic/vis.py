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
import dolfinx as df

from domain_vars import fest_lost_2d_domain
from process.mechanic.gnd import base_mapping, get_einspannung_2d_fem, get_ensemble_einspannung_2d_fem, get_sigma_fem


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
    predictions = predictions  * scale.U


    # GROUND
    
    comm = MPI.COMM_WORLD
    GROUND = get_einspannung_2d_fem(fest_lost_2d_domain)
    #GROUND = load_fem_results("BASELINE/mechanic/2d/ground_truth.npy")
    
    domain = GROUND.function_space.mesh
    _perform_eval, eval_points_3d, bb_tree = initialize_point_evaluation(
        domain, points, comm
    )

    ### FIXED EVAL
    V = GROUND.function_space
    V_x, dof_map_x = V.sub(0).collapse()
    V_y, dof_map_y = V.sub(1).collapse()

    u_x_func = df.fem.Function(V_x)
    u_y_func = df.fem.Function(V_y)

    u_x_func.x.array[:] = GROUND.x.array[dof_map_x]
    u_y_func.x.array[:] = GROUND.x.array[dof_map_y]

    gt_u_x_flat = evaluate_solution_at_points_on_rank_0(u_x_func, eval_points_3d, bb_tree, domain, comm)
    gt_u_y_flat = evaluate_solution_at_points_on_rank_0(u_y_func, eval_points_3d, bb_tree, domain, comm)

    if comm.rank == 0:
        ground_values_at_points = np.hstack((gt_u_x_flat[:, np.newaxis], gt_u_y_flat[:, np.newaxis]))
    else:
        ground_values_at_points = None
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
    scale_factor = 50
    
    # Predicted Deformed Shape
    ax = axes[3, 0]
    deformed_X_pred = X + scale_factor * pred_u_x_2d
    deformed_Y_pred = Y + scale_factor * pred_u_y_2d
    ax.scatter(X, Y, c='blue', s=0.5, alpha=0.3, label='Original')
    ax.scatter(deformed_X_pred, deformed_Y_pred, c='red', s=0.5, alpha=0.7, label=f'Deformed (×{scale_factor})')
    ax.set_title(f"Predicted Deformed Shape, with scale: {scale_factor}")
    

    # Ground Truth Deformed Shape
    ax = axes[3, 1]
    deformed_X_gt = X + scale_factor * gt_u_x_2d
    deformed_Y_gt = Y + scale_factor * gt_u_y_2d
    ax.scatter(X, Y, c='blue', s=0.5, alpha=0.3, label='Original')
    ax.scatter(deformed_X_gt, deformed_Y_gt, c='green', s=0.5, alpha=0.7, label=f'Deformed (×{scale_factor})')
    ax.set_title(f"Ground Truth Deformed Shape, with scale: {scale_factor}")
    

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
    
    return {'fig': fig}


def vis_2d_ensemble(model, scale: Scale, **kwargs):

    # Get domain and evaluation points
    x_min, x_max = fest_lost_2d_domain.spatial['x']
    y_min, y_max = fest_lost_2d_domain.spatial['y']
    nx, ny = kwargs.get('resolution', (100, 50))
    
    x_points = np.linspace(x_min, x_max, nx)
    y_points = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x_points, y_points)
    points = np.vstack((X.flatten(), Y.flatten())).T
    scaled_points = points / np.array([scale.L, scale.L])

    # Get ensemble predictions [u_x, u_y, sigma_x, sigma_y, tau_xy]
    predictions = model.predict(scaled_points)
    
    # Scale predictions back to physical units
    pred_fields = {
        'u_x': predictions[:, 0] * scale.U,
        'u_y': predictions[:, 1] * scale.U,
        'sigma_x': predictions[:, 2] * scale.sigma,
        'sigma_y': predictions[:, 3] * scale.sigma,
        'tau_xy': predictions[:, 4] * scale.sigma
    }

    comm = MPI.COMM_WORLD

    # Get comprehensive FEM solution using enhanced helper system
    try:
        # Try cached ensemble results
        cached_path = "BASELINE/mechanic/2d/ensemble_ground_truth.npy"
        ensemble_cache = load_fem_results(cached_path)
        
        if ensemble_cache is None:
            raise FileNotFoundError("Ensemble cache not found")
            
        print("Loaded cached ensemble FEM results")
        gt_fields = {
            'u_x': ensemble_cache[:, 0],
            'u_y': ensemble_cache[:, 1],
            'sigma_x': ensemble_cache[:, 2],
            'sigma_y': ensemble_cache[:, 3],
            'tau_xy': ensemble_cache[:, 4]
        }
        
    except (FileNotFoundError, Exception):
        print("Computing ensemble FEM solution using enhanced helpers...")
        
        # Get comprehensive ensemble solution
        ensemble = get_ensemble_einspannung_2d_fem(fest_lost_2d_domain)
        
        # Initialize point evaluation
        domain = ensemble['displacement'].function_space.mesh
        perform_eval, eval_points_3d, bb_tree = initialize_point_evaluation(
            domain, points, comm
        )
        
        if not perform_eval:
            print("Point evaluation setup failed")
            return {'field': None}
        
        # Evaluate displacement fields
        V = ensemble['displacement'].function_space
        V_x, dof_map_x = V.sub(0).collapse()
        V_y, dof_map_y = V.sub(1).collapse()

        u_x_func = df.fem.Function(V_x)
        u_y_func = df.fem.Function(V_y)
        u_x_func.x.array[:] = ensemble['displacement'].x.array[dof_map_x]
        u_y_func.x.array[:] = ensemble['displacement'].x.array[dof_map_y]

        # Evaluate all fields using helper functions
        field_evaluations = {}
        
        # Displacement components
        field_evaluations['u_x'] = evaluate_solution_at_points_on_rank_0(
            u_x_func, eval_points_3d, bb_tree, domain, comm
        )
        field_evaluations['u_y'] = evaluate_solution_at_points_on_rank_0(
            u_y_func, eval_points_3d, bb_tree, domain, comm
        )
        
        # Stress components
        for stress_name in ['sigma_xx', 'sigma_yy', 'tau_xy']:
            field_evaluations[stress_name] = evaluate_solution_at_points_on_rank_0(
                ensemble[stress_name], eval_points_3d, bb_tree, domain, comm
            )
        
        if comm.rank == 0:
            # Process and flatten results
            gt_fields = {}
            for key, eval_result in field_evaluations.items():
                if eval_result is not None:
                    gt_fields[key.replace('_xx', '_x').replace('_yy', '_y')] = (
                        eval_result.flatten() if eval_result.ndim > 1 else eval_result
                    )
            
            # Rename for consistency
            if 'sigma_x' not in gt_fields and 'sigma_xx' in gt_fields:
                gt_fields['sigma_x'] = gt_fields.pop('sigma_xx')
            if 'sigma_y' not in gt_fields and 'sigma_yy' in gt_fields:
                gt_fields['sigma_y'] = gt_fields.pop('sigma_yy')
            
            # Cache ensemble results
            ensemble_cache = np.column_stack([
                gt_fields['u_x'], gt_fields['u_y'], 
                gt_fields['sigma_x'], gt_fields['sigma_y'], gt_fields['tau_xy']
            ])
            save_fem_results(cached_path, ensemble_cache)
            
        else:
            return {'field': None}

    # Create comprehensive ensemble visualization
    return _create_ensemble_visualization(
        X, Y, points, pred_fields, gt_fields, 
        x_min, x_max, y_min, y_max, nx, ny
    )

def _create_ensemble_visualization(X, Y, points, pred_fields, gt_fields, 
                                 x_min, x_max, y_min, y_max, nx, ny):
    """
    Helper function to create comprehensive ensemble visualization
    """
    # Check array consistency
    n_points = len(pred_fields['u_x'])
    for key in gt_fields:
        if len(gt_fields[key]) != n_points:
            print(f"Warning: Size mismatch for {key}")
            min_size = min(len(pred_fields[key]), len(gt_fields[key]))
            pred_fields[key] = pred_fields[key][:min_size]
            gt_fields[key] = gt_fields[key][:min_size]

    # Create comprehensive visualization
    fig, axes = plt.subplots(5, 3, figsize=(20, 32))
    fig.suptitle('Enhanced Ensemble Field Visualization: Prediction vs. Ground Truth', fontsize=22)

    # Field configuration
    field_config = [
        {'key': 'u_x', 'name': 'X-Displacement ($u_x$)', 'unit': '[m]', 'cmap': 'RdBu_r'},
        {'key': 'u_y', 'name': 'Y-Displacement ($u_y$)', 'unit': '[m]', 'cmap': 'RdBu_r'},
        {'key': 'sigma_x', 'name': 'X-Stress ($\\sigma_x$)', 'unit': '[Pa]', 'cmap': 'coolwarm'},
        {'key': 'sigma_y', 'name': 'Y-Stress ($\\sigma_y$)', 'unit': '[Pa]', 'cmap': 'coolwarm'},
        {'key': 'tau_xy', 'name': 'Shear Stress ($\\tau_{xy}$)', 'unit': '[Pa]', 'cmap': 'seismic'}
    ]

    for i, config in enumerate(field_config):
        key = config['key']
        
        try:
            # Reshape data for contour plots
            pred_2d = pred_fields[key][:nx*ny].reshape(ny, nx)
            gt_2d = gt_fields[key][:nx*ny].reshape(ny, nx)
            error_2d = (pred_fields[key][:nx*ny] - gt_fields[key][:nx*ny]).reshape(ny, nx)
            
            # Plot predicted field
            ax = axes[i, 0]
            contour = ax.contourf(X, Y, pred_2d, levels=20, cmap=config['cmap'])
            ax.set_title(f"Predicted {config['name']}")
            cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
            cbar.set_label(config['unit'])

            # Plot ground truth field
            ax = axes[i, 1]
            contour_gt = ax.contourf(X, Y, gt_2d, levels=20, cmap=config['cmap'])
            ax.set_title(f"Ground Truth {config['name']}")
            cbar_gt = plt.colorbar(contour_gt, ax=ax, shrink=0.8)
            cbar_gt.set_label(config['unit'])

            # Plot error field
            ax = axes[i, 2]
            contour_err = ax.contourf(X, Y, error_2d, levels=20, cmap='PRGn')
            ax.set_title(f"{config['name']} Error")
            cbar_err = plt.colorbar(contour_err, ax=ax, shrink=0.8)
            cbar_err.set_label(f"Error {config['unit']}")
            
        except Exception as e:
            print(f"Error plotting field {key}: {e}")
            for j in range(3):
                axes[i, j].text(0.5, 0.5, f"Error plotting {key}", 
                              ha='center', va='center', transform=axes[i, j].transAxes)

    # Format all plots
    _format_visualization_axes(axes, x_min, x_max, y_min, y_max)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    return {'field': fig}

def _format_visualization_axes(axes, x_min, x_max, y_min, y_max):
    """
    Helper function to format visualization axes consistently
    """
    beam_outline = np.array([
        [x_min, y_min], [x_max, y_min], 
        [x_max, y_max], [x_min, y_max], [x_min, y_min]
    ])
    
    for ax in axes.flat:
        if ax.get_title():  # Skip empty subplots
            ax.plot(beam_outline[:, 0], beam_outline[:, 1], 'k-', linewidth=1.5, alpha=0.9)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_xlabel('X-coordinate [m]')
            ax.set_ylabel('Y-coordinate [m]')
            ax.set_aspect('equal')
            
            # Set limits with buffer
            buffer_x = 0.1 * (x_max - x_min)
            buffer_y = 0.1 * (y_max - y_min)
            ax.set_xlim(x_min - buffer_x, x_max + buffer_x)
            ax.set_ylim(y_min - buffer_y, y_max + buffer_y)
