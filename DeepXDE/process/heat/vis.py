import os
from utils.metadata import Domain
from process.moisture.scale import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm

from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as mcolors
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
    # --- Points ---
    scaled_points = points_data['spacetime_points_flat'] / np.array([scale.L, scale.L, scale.t])
    X_plot, Y_plot = points_data['spacetime_meshgrid']['ij']['x'], points_data['spacetime_meshgrid']['ij']['y']
    t = points_data['temporal_coords']['t']

    # --- Get Predictions ---
    predictions = model.predict(scaled_points)
    predictions = points_data['reshape_utils']['pred_to_ij'](predictions)
    predictions = predictions * scale.T

    fields = vis_transient_field_test(predictions, fem_value_points, X_plot, Y_plot, t, **kwargs)
    test_slice = visualize_transient_spacetime_slice(predictions, fem_value_points, X_plot, Y_plot, t, **kwargs)
    return {**fields, **test_slice}



def vis_transient_field_test(predictions, fem_value_points, X, Y, t, **kwargs):
    difference = predictions - fem_value_points

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    combined_min = min(np.nanmin(predictions), np.nanmin(fem_value_points))
    combined_max = max(np.nanmax(predictions), np.nanmax(fem_value_points))
    
    # For difference: use symmetric scale around zero
    diff_abs_max = np.nanmax(np.abs(difference))
    diff_vmin, diff_vmax = -diff_abs_max, diff_abs_max
    
    # Handle NaN cases
    #if np.isnan(pred_vmin): pred_vmin, pred_vmax = 0, 1
    #if np.isnan(ground_vmin): ground_vmin, ground_vmax = 0, 1
    #if np.isnan(diff_vmin): diff_vmin, diff_vmax = -1, 1
    # --- 4. Initial Plot Frame ---
    pred_plot = predictions[:,:,0]
    fem_plot = fem_value_points[:,:,0]
    diff_plot = difference[:,:,0]
    
    cont1 = axes[0].contourf(X[:,:,0], Y[:,:,0], pred_plot, 50, cmap=cm.jet, vmin=0, vmax=100)
    cont2 = axes[1].contourf(X[:,:,0], Y[:,:,0], fem_plot, 50, cmap=cm.jet, vmin=0, vmax=100)
    cont3 = axes[2].contourf(X[:,:,0], Y[:,:,0], diff_plot, 50, cmap=cm.RdBu_r, vmin=diff_vmin, vmax=diff_vmax)
    
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
            for c in ax.collections:
                c.remove()

        pred_plot = predictions[:,:,frame]
        fem_plot = fem_value_points[:,:,frame]
        diff_plot = difference[:,:,frame]
        
        cont1 = axes[0].contourf(X[:,:,frame], Y[:,:,frame], pred_plot, 50, cmap=cm.jet, vmin=0, vmax=100)
        cont2 = axes[1].contourf(X[:,:,frame], Y[:,:,frame], fem_plot, 50, cmap=cm.jet, vmin=0, vmax=100)
        cont3 = axes[2].contourf(X[:,:,frame], Y[:,:,frame], diff_plot, 50, cmap=cm.RdBu_r, vmin=diff_vmin, vmax=diff_vmax)
        
        # Redraw colorbars in each frame
        cbar1.mappable.set_clim(0, 100)
        cbar2.mappable.set_clim(0, 100)
        cbar3.mappable.set_clim(diff_vmin, diff_vmax)

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

def visualize_transient_spacetime_slice(predictions, fem_value_points, X, Y, t, slice_type='middle', **kwargs):
    """
    Create space-time heatmaps by taking 1D slices through the domain
    """
    if slice_type == 'middle':
        print("Using middle slice for visualization")
        mid_y = predictions.shape[1] // 2
        print(f"Middle y index: {mid_y}")
        pred_slice = predictions[:, mid_y, :]  # (x, t)
        fem_slice = fem_value_points[:, mid_y, :]
        x_coords = X[:, mid_y, 0]
        slice_label = f'y = {Y[0, mid_y, 0]:.2f}m'
    elif slice_type == 'diagonal':
        print("Using diagonal slice for visualization")
        min_dim = min(predictions.shape[0], predictions.shape[1])
        pred_slice = predictions[range(min_dim), range(min_dim), :]
        fem_slice = fem_value_points[range(min_dim), range(min_dim), :]
        x_coords = np.sqrt(X[range(min_dim), range(min_dim), 0]**2 + Y[range(min_dim), range(min_dim), 0]**2)
        slice_label = 'Diagonal'
    
    difference_slice = pred_slice - fem_slice
    
    # Convert time to days
    t_days = t / (60*60*24)
    
    # Create meshgrids for plotting
    T_mesh, X_mesh = np.meshgrid(t_days, x_coords)
    

    # Get symmetric range for difference
    diff_max = np.nanmax(np.abs(difference_slice))
    diff_norm = TwoSlopeNorm(vmin=-diff_max, vcenter=0, vmax=diff_max)
    
    # === COMBINED PLOT ===
    fig_combined, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot heatmaps
    im1 = axes[0].contourf(T_mesh, X_mesh, pred_slice, 50, cmap=cm.jet)
    im2 = axes[1].contourf(T_mesh, X_mesh, fem_slice, 50, cmap=cm.jet)
    im3 = axes[2].contourf(T_mesh, X_mesh, difference_slice, 50, cmap=cm.RdBu_r, norm=diff_norm)
    
    # Add colorbars
    fig_combined.colorbar(im1, ax=axes[0], label='Temperature [°C]')
    fig_combined.colorbar(im2, ax=axes[1], label='Temperature [°C]')
    fig_combined.colorbar(im3, ax=axes[2], label='Difference [°C]')
    
    # Labels and titles
    for ax in axes:
        ax.set_xlabel('Time [days]')
        ax.set_ylabel('Position [m]')
    
    #axes[0].set_title(f'Prediction - {slice_label}')
    #axes[1].set_title(f'Ground Truth - {slice_label}')
    #axes[2].set_title(f'Difference - {slice_label}')
    
    plt.tight_layout()
    
    # === INDIVIDUAL PLOTS ===
    # Prediction plot
    fig_pred, ax_pred = plt.subplots(figsize=(8, 6))
    im_pred = ax_pred.contourf(T_mesh, X_mesh, pred_slice, 50, cmap=cm.jet)
    fig_pred.colorbar(im_pred, ax=ax_pred, label='Temperature [°C]')
    ax_pred.set_xlabel('Time [days]')
    ax_pred.set_ylabel('Position [m]')
    ax_pred.set_title(f'Prediction - {slice_label}')
    plt.tight_layout()
    
    # Ground truth plot
    fig_ground, ax_ground = plt.subplots(figsize=(8, 6))
    im_ground = ax_ground.contourf(T_mesh, X_mesh, fem_slice, 50, cmap=cm.jet)
    fig_ground.colorbar(im_ground, ax=ax_ground, label='Temperature [°C]')
    ax_ground.set_xlabel('Time [days]')
    ax_ground.set_ylabel('Position [m]')
    ax_ground.set_title(f'Ground Truth - {slice_label}')
    plt.tight_layout()
    
    # Difference plot
    fig_diff, ax_diff = plt.subplots(figsize=(8, 6))
    im_diff = ax_diff.contourf(T_mesh, X_mesh, difference_slice, 50, cmap=cm.RdBu_r, norm=diff_norm)
    fig_diff.colorbar(im_diff, ax=ax_diff, label='Difference [°C]')
    ax_diff.set_xlabel('Time [days]')
    ax_diff.set_ylabel('Position [m]')
    ax_diff.set_title(f'Difference - {slice_label}')
    plt.tight_layout()
    
    return {
        'slice': fig_combined,
        'slice_field_pred': fig_pred,
        'slice_field_ground': fig_ground,
        'slice_div': fig_diff
    }
