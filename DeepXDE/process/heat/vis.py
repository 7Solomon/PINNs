import os
from utils.metadata import Domain
from process.moisture.scale import *
from vis import get_2d_domain, get_2d_time_domain
from domain_vars import transient_heat_2d_domain, steady_heat_2d_domain
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm

from process.heat.scale import *
def visualize_steady_field(model, **kwargs):
    scale = Scale(steady_heat_2d_domain)
    domain = get_2d_domain(steady_heat_2d_domain, scale)
    points, X, Y, nx, ny = domain['normal']
    points_scaled, X_scaled, Y_scaled, nx, ny = domain['scaled']

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

def visualize_transient_field(model, **kwargs):
    scale = Scale(transient_heat_2d_domain)
    domain = get_2d_time_domain(transient_heat_2d_domain, scale)

    points, X, Y, t, nx, ny, nt = domain['normal']
    scaled_points, X_scaled, Y_scaled, t_scaled, nx, ny, nt = domain['scaled']
    
    print('scaled_points: ', scaled_points.shape)
    print('X_scaled', X_scaled.shape)
    print('Y_scaled', Y_scaled.shape)
    predictions = model.predict(scaled_points)
    predictions = predictions.reshape(ny, nx, nt)
    predictions = predictions * scale.T

    fig, ax = plt.subplots(figsize=(10, 5))

    
    # First frame
    cont = ax.contourf(X[:,:,0], Y[:,:,0], predictions[:,:,0], 50, cmap=cm.jet, vmin=predictions.min(), vmax=predictions.max())
    cbar = fig.colorbar(cont, ax=ax)
    cbar.set_label('Field Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Distribution at [t={(t[0,0,0]/(60*60*24)):.3f} days]')
    
    def update(frame):
        ax.clear()
        cont = ax.contourf(X[:,:,frame], Y[:,:,frame], predictions[:,:,frame], 50, cmap=cm.jet, vmin=predictions.min(), vmax=predictions.max())
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Distribution at [t={(t[0,0,frame]/(60*60*24)):.3f} days]')
        cbar.update_normal(cont)
        return cont
    
    ani = animation.FuncAnimation(fig, update, frames=nt, interval=100)

    plt.tight_layout()
    plt.show()
    #return {'field': fig}
    return {'field': ani, 'fig': fig}


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