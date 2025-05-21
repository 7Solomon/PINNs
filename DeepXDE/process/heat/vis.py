import os
from process.moisture.scale import *
from vis import get_2d_domain, get_2d_time_domain

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm

from process.heat.scale import *
def visualize_steady_field(model, domain_variabels):
    domain = get_2d_domain(domain_variabels, scale_x, scale_y)
    points, X, Y, nx, ny = domain['normal']
    points_scaled, X_scaled, Y_scaled, nx, ny = domain['scaled']

    predictions = model.predict(points_scaled)
    Z = predictions.reshape(ny, nx)

    #PLOT
    plt.contourf(X, Y, Z, 50, cmap=cm.jet)
    plt.colorbar(label='Steady Heat')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Predicted')
    plt.tight_layout()
    #plt.savefig('heat_field.png', dpi=300)
    plt.show()

def visualize_transient_field(model, domain_variabels,save_animation=False):
    domain = get_2d_time_domain(domain_variabels, scale_x, scale_y, scale_t)
    
    points, X, Y, t, nx, ny, nt = domain['normal']
    scaled_points, X_scaled, Y_scaled, t_scaled, nx, ny, nt = domain['scaled']

    predictions = model.predict(scaled_points)
    predictions = rescale_value(predictions)

    fig, ax = plt.subplots(figsize=(8, 6))

    
    # First frame
    cont = ax.contourf(X, Y, predictions[:,:,0], 50, cmap=cm.jet, vmin=predictions.min(), vmax=predictions.max())
    cbar = fig.colorbar(cont, ax=ax)
    cbar.set_label('Field Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Distribution at [t={t[0]:.3f}]')
    
    def update(frame):
        ax.clear()
        cont = ax.contourf(X, Y, predictions[:,:,frame], 50, cmap=cm.jet, vmin=predictions.min(), vmax=predictions.max())
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Distribution[t={t[frame]:.3f}]')
        cbar.update_normal(cont)
        return cont
    
    ani = animation.FuncAnimation(fig, update, frames=nt, interval=100)
    
    if save_animation:
        ani.save('ka_volution.mp4', writer='ffmpeg', dpi=300)
        
    plt.tight_layout()
    plt.show()


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