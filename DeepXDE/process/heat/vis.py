import os
from points import get_meshgrid_for_visualization
from FEM.output import load_fem_results, save_fem_results
from process.heat.gnd import get_transient_fem, get_transient_fem_points
from utils.metadata import Domain
from process.moisture.scale import *
from domain_vars import transient_heat_2d_domain, steady_heat_2d_domain
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm


from mpi4py import MPI

from process.heat.scale import *
def visualize_steady_field(model, scale: Scale, points_data: dict, **kwargs):
    # Create grid
    nx, ny = points_data['resolution']['x'], points_data['resolution']['y']
    x = points_data['spatial_coords']['x']
    y = points_data['spatial_coords']['y']
    X, Y = points_data['spatial_meshgrid']['x'], points_data['spatial_meshgrid']['y']

    scaled_X = X.copy() / scale.L
    scaled_Y = Y.copy() / scale.L
    points_scaled = np.vstack((scaled_X.flatten(), scaled_Y.flatten())).T

    predictions = model.predict(points_scaled)
    predictions = predictions.reshape(nx, ny)
    predictions = predictions * scale.T

    #PLOT
    plt.figure(figsize=(10,5))
    plt.contourf(X, Y, predictions, 50, cmap=cm.jet)
    plt.colorbar(label='Steady Heat')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Predicted')
    plt.tight_layout()
    #plt.savefig('heat_field.png', dpi=300)
    #plt.show()
    return {'field': plt.gcf()}

def visualize_transient_field(model, scale: Scale, points_data: dict, fem_value_points: np.ndarray, **kwargs):
    t = points_data['temporal_coords']['t']
    
    scaled_points = points_data['spacetime_points_flat'] / np.array([scale.L, scale.L, scale.t])
    # Use the 'ij' meshgrid for matplotlib plotting
    X_plot, Y_plot = points_data['spacetime_meshgrid']['xy']['x'], points_data['spacetime_meshgrid']['xy']['y']

    # --- 2. Get Predictions ---
    predictions = model.predict(scaled_points)
    print(f"Predictions shape: {predictions.shape}")  # Debugging line

    # Reshape predictions back to an 'ij' grid (nx, ny, nt)
    predictions = points_data['reshape_utils']['pred_to_ij'](predictions)
    print(f"Reshaped predictions shape: {predictions.shape}")  # Debugging line
    predictions = predictions * scale.T

    difference = predictions - fem_value_points

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Determine consistent color bar limits
    pred_vmin, pred_vmax = np.nanmin(predictions), np.nanmax(predictions)
    ground_vmin, ground_vmax = np.nanmin(fem_value_points), np.nanmax(fem_value_points)
    diff_vmin, diff_vmax = np.nanmin(difference), np.nanmax(difference)

    # Handle cases where data might be all NaN
    ground_vmin = 0 if np.isnan(ground_vmin) else ground_vmin
    ground_vmax = 1 if np.isnan(ground_vmax) else ground_vmax
    diff_vmin = -1 if np.isnan(diff_vmin) else diff_vmin
    diff_vmax = 1 if np.isnan(diff_vmax) else diff_vmax

    # --- 4. Initial Plot Frame ---
    pred_plot = predictions[:,:,0].T
    fem_plot = fem_value_points[:,:,0].T
    diff_plot = difference[:,:,0].T
    
    cont1 = axes[0].contourf(X_plot[:,:,0], Y_plot[:,:,0], pred_plot, 50, cmap=cm.jet, vmin=pred_vmin, vmax=pred_vmax)
    cont2 = axes[1].contourf(X_plot[:,:,0], Y_plot[:,:,0], fem_plot, 50, cmap=cm.jet, vmin=ground_vmin, vmax=ground_vmax)
    cont3 = axes[2].contourf(X_plot[:,:,0], Y_plot[:,:,0], diff_plot, 50, cmap=cm.RdBu_r, vmin=diff_vmin, vmax=diff_vmax)
    
    cbar1 = fig.colorbar(cont1, ax=axes[0])
    cbar2 = fig.colorbar(cont2, ax=axes[1])
    cbar3 = fig.colorbar(cont3, ax=axes[2])
    
    axes[0].set_title(f'Prediction at t={(t[0]/(60*60*24)):.3f} days')
    axes[1].set_title(f'Ground Truth at t={(t[0]/(60*60*24)):.3f} days')
    axes[2].set_title(f'Difference at t={(t[0]/(60*60*24)):.3f} days')


    
    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

    # --- 5. Animation Update Function ---
    def update(frame):
        for ax in axes:
            ax.clear()

        pred_plot = predictions[:,:,frame].T
        fem_plot = fem_value_points[:,:,frame].T
        diff_plot = difference[:,:,frame].T
        
        cont1 = axes[0].contourf(X_plot[:,:,frame], Y_plot[:,:,frame], pred_plot, 50, cmap=cm.jet, vmin=pred_vmin, vmax=pred_vmax)
        cont2 = axes[1].contourf(X_plot[:,:,frame], Y_plot[:,:,frame], fem_plot, 50, cmap=cm.jet, vmin=ground_vmin, vmax=ground_vmax)
        cont3 = axes[2].contourf(X_plot[:,:,frame], Y_plot[:,:,frame], diff_plot, 50, cmap=cm.RdBu_r, vmin=diff_vmin, vmax=diff_vmax)
        
        axes[0].set_title(f'Prediction at t={(t[frame]/(60*60*24)):.3f} days')
        axes[1].set_title(f'Ground Truth at t={(t[frame]/(60*60*24)):.3f} days')
        axes[2].set_title(f'Difference at t={(t[frame]/(60*60*24)):.3f} days')
        
        for ax in axes:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
        
        # No need to update colorbars if vmin/vmax are fixed
        return [cont1, cont2, cont3]
    
    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=100,
                                  repeat=True)
    plt.tight_layout()
    
    return {'field': ani, 'fig': fig}#, 'difference': difference}
def test_vis_ground_fixed(points_data, fem_value_points):
    # Create grid
    meshgrid = get_meshgrid_for_visualization(points_data, indexing='xy')
    X, Y = meshgrid['x'], meshgrid['y']
    t = points_data['temporal_coords']['t']
    
    # VAl
    pred_plot = fem_value_points[:,:,0] 
    
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    pred_vmin, pred_vmax = np.nanmin(fem_value_points), np.nanmax(fem_value_points)

    # First frame
    cont1 = axes.contourf(X[:,:,0], Y[:,:,0], pred_plot.T, 50, cmap=cm.jet, vmin=pred_vmin, vmax=pred_vmax)
    
    # Colorbars
    cbar1 = fig.colorbar(cont1, ax=axes)

    # Labels
    axes.set_title(f'Ground Truth at t={(t[0]/(60*60*24)):.3f} days')
    axes.set_xlabel('x')
    axes.set_ylabel('y')

    def update(frame):
        axes.clear()
        
        # Transpose for plotting
        pred_plot = fem_value_points[:,:,frame]
        
        cont1 = axes.contourf(X[:,:,frame], Y[:,:,frame], pred_plot.T, 50, cmap=cm.jet, vmin=pred_vmin, vmax=pred_vmax)
        
        axes.set_title(f'Ground Truth at t={(t[frame]/(60*60*24)):.3f} days')
        axes.set_xlabel('x')
        axes.set_ylabel('y')
        cbar1.update_normal(cont1)
        
        return [cont1]

    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=100)
    plt.tight_layout()
    
    return {'field': ani, 'fig': fig}