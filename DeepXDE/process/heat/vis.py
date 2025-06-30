import os
from FEM.output import load_fem_results, save_fem_results
from process.heat.gnd import get_transient_fem
from utils.metadata import Domain
from process.moisture.scale import *
from domain_vars import transient_heat_2d_domain, steady_heat_2d_domain
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm


from mpi4py import MPI

from process.heat.scale import *
def visualize_steady_field(model, scale: Scale, **kwargs):
    
    min_x, max_x = steady_heat_2d_domain.spatial['x']
    min_y, max_y = steady_heat_2d_domain.spatial['y']
    # Create grid
    nx, ny = 100, 50
    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    X, Y = np.meshgrid(x, y)
    scaled_X = X.copy() / scale.L
    scaled_Y = Y.copy() / scale.L
    points_scaled = np.vstack((scaled_X.flatten(), scaled_Y.flatten())).T

    predictions = model.predict(points_scaled)
    predictions = predictions.reshape(ny, nx)
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

def visualize_transient_field(model, scale: Scale, **kwargs):
    min_x, max_x = transient_heat_2d_domain.spatial['x']
    min_y, max_y = transient_heat_2d_domain.spatial['y']
    min_t, max_t = transient_heat_2d_domain.temporal['t']


    # Create grid
    nx, ny, nt = 100, 50, 100
    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    t = np.linspace(min_t, max_t, nt)
    X, Y, T = np.meshgrid(x, y, t)

    scaled_X = X.copy() / scale.L
    scaled_Y = Y.copy() / scale.L
    scaled_T = T.copy() / scale.t

    scaled_points = np.vstack((scaled_X.flatten(), scaled_Y.flatten(), scaled_T.flatten())).T

    
    predictions = model.predict(scaled_points)
    predictions = predictions.reshape(ny, nx, nt)
    predictions = predictions * scale.T


    # Ground
    X_spatial_grid, Y_spatial_grid = np.meshgrid(x, y, indexing='xy') # ny, nx
    evaluation_spatial_points_xy = np.stack((X_spatial_grid.ravel(), Y_spatial_grid.ravel()), axis=-1)


    #_, ground_eval_flat_time_spatial = get_transient_fem(
    #    transient_heat_2d_domain,
    #    evaluation_times=t,
    #    evaluation_spatial_points_xy=evaluation_spatial_points_xy
    #)
    #ground_truth = np.full((ny, nx, nt), np.nan) 
    #ground_truth = ground_eval_flat_time_spatial.reshape(nt, ny, nx).transpose(1, 2, 0)
    #save_fem_results("BASELINE/heat/ground_truth.npy", ground_truth)
    ground_truth = load_fem_results("BASELINE/heat/ground_truth.npy")



    difference = predictions - ground_truth

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # BAR
    pred_vmin, pred_vmax = np.nanmin(predictions), np.nanmax(predictions)
    ground_vmin, ground_vmax = np.nanmin(ground_truth), np.nanmax(ground_truth)
    diff_vmin, diff_vmax = np.nanmin(difference), np.nanmax(difference)
    
    ground_vmin = 0 if np.isnan(ground_vmin) else ground_vmin
    ground_vmax = 1 if np.isnan(ground_vmax) else ground_vmax
    diff_vmin = -1 if np.isnan(diff_vmin) else diff_vmin
    diff_vmax = 1 if np.isnan(diff_vmax) else diff_vmax
    
    # First frame
    cont1 = axes[0].contourf(X[:,:,0], Y[:,:,0], predictions[:,:,0], 50, cmap=cm.jet, vmin=pred_vmin, vmax=pred_vmax)
    cont2 = axes[1].contourf(X[:,:,0], Y[:,:,0], ground_truth[:,:,0], 50, cmap=cm.jet, vmin=ground_vmin, vmax=ground_vmax)
    cont3 = axes[2].contourf(X[:,:,0], Y[:,:,0], difference[:,:,0], 50, cmap=cm.RdBu_r, vmin=diff_vmin, vmax=diff_vmax)
    
    # Colorbars
    cbar1 = fig.colorbar(cont1, ax=axes[0])
    cbar2 = fig.colorbar(cont2, ax=axes[1])
    cbar3 = fig.colorbar(cont3, ax=axes[2])
    
    # Labels
    axes[0].set_title(f'Prediction at t={(t[0]/(60*60*24)):.3f} days')
    axes[1].set_title(f'Ground Truth at t={(t[0]/(60*60*24)):.3f} days')
    axes[2].set_title(f'Difference at t={(t[0]/(60*60*24)):.3f} days')
    
    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    def update(frame):
        for ax in axes:
            ax.clear()
        
        cont1 = axes[0].contourf(X[:,:,frame], Y[:,:,frame], predictions[:,:,frame], 50, cmap=cm.jet, vmin=pred_vmin, vmax=pred_vmax)
        cont2 = axes[1].contourf(X[:,:,frame], Y[:,:,frame], ground_truth[:,:,frame], 50, cmap=cm.jet, vmin=ground_vmin, vmax=ground_vmax)
        cont3 = axes[2].contourf(X[:,:,frame], Y[:,:,frame], difference[:,:,frame], 50, cmap=cm.RdBu_r, vmin=diff_vmin, vmax=diff_vmax)
        
        axes[0].set_title(f'Prediction at t={(t[frame]/(60*60*24)):.3f} days')
        axes[1].set_title(f'Ground Truth at t={(t[frame]/(60*60*24)):.3f} days')
        axes[2].set_title(f'Difference at t={(t[frame]/(60*60*24)):.3f} days')
        
        for ax in axes:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        
        cbar1.update_normal(cont1)
        cbar2.update_normal(cont2)
        cbar3.update_normal(cont3)
        
        return [cont1, cont2, cont3]
    
    ani = animation.FuncAnimation(fig, update, frames=nt, interval=100)
    plt.tight_layout()
    
    return {'field': ani, 'fig': fig}#, 'difference': difference}


#def visualize_field(model, type, inverse_scale=None):
#    if type == 'steady':
#        domain_stuff = {
#            'x_min': 0,
#            'x_max': 2,
#            'y_min': 0,
#            'y_max': 1,
#            'min_val': 0.0,
#            'max_val': 100.0
#        }
#        visualize_steady_field(model, domain_stuff, inverse_scale=inverse_scale)
#    elif type == 'transient':
#        domain_stuff = {
#            'x_min': 0,
#            'x_max': 2,
#            'y_min': 0,
#            'y_max': 1,
#            't_min': 0,
#            't_max': 1.1e7,
#            'min_val': 0.0,
#            'max_val': 100.0
#        }
#        visualize_time_dependent_field(model, domain_stuff,inverse_scale=inverse_scale, animate=True)
#
def visualize_divergence(model, subtype, inverse_scale=None):
    raise NotImplementedError("Divergence visualization is not implemented yet.")
    #from utils.COMSOL import load_COMSOL_file_data, analyze_transient_COMSOL_file_data, analyze_steady_COMSOL_file_data
    #if subtype == 'steady':
    #    from vis import vis_steady_diffrence
    #    file_data = load_COMSOL_file_data(os.path.join('BASELINE', 'heat', '2d_steady.txt'))
    #    domain, temp = analyze_steady_COMSOL_file_data(file_data)
#
    #    vis_steady_diffrence(model, domain, temp, inverse_scale=inverse_scale)
    #elif subtype == 'transient':
    #    from vis import vis_time_diffrence
    #    file_data = load_COMSOL_file_data(os.path.join('BASELINE', 'heat', '2d_transient.txt'))
    #    coords2d, times, temp2d = analyze_transient_COMSOL_file_data(file_data)
    #    vis_time_diffrence(model, coords2d, times, temp2d, inverse_scale, animate=True)