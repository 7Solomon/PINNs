import os
from vis import visualize_time_dependent_field, visualize_steady_field
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
    
    #return Z
def visualize_field(model, type, inverse_scale=None):
    if type == 'steady':
        domain_stuff = {
            'x_min': 0,
            'x_max': 2,
            'y_min': 0,
            'y_max': 1,
            'min_val': 0.0,
            'max_val': 100.0
        }
        visualize_steady_field(model, domain_stuff, inverse_scale=inverse_scale)
    elif type == 'transient':
        domain_stuff = {
            'x_min': 0,
            'x_max': 2,
            'y_min': 0,
            'y_max': 1,
            't_min': 0,
            't_max': 1.1e7,
            'min_val': 0.0,
            'max_val': 100.0
        }
        visualize_time_dependent_field(model, domain_stuff,inverse_scale=inverse_scale, animate=True)

def visualize_divergence(model, subtype, inverse_scale=None):
    from utils.COMSOL import load_COMSOL_file_data, analyze_transient_COMSOL_file_data, analyze_steady_COMSOL_file_data
    if subtype == 'steady':
        from vis import vis_steady_diffrence
        file_data = load_COMSOL_file_data(os.path.join('BASELINE', 'heat', '2d_steady.txt'))
        domain, temp = analyze_steady_COMSOL_file_data(file_data)

        vis_steady_diffrence(model, domain, temp, inverse_scale=inverse_scale)
    elif subtype == 'transient':
        from vis import vis_time_diffrence
        file_data = load_COMSOL_file_data(os.path.join('BASELINE', 'heat', '2d_transient.txt'))
        coords2d, times, temp2d = analyze_transient_COMSOL_file_data(file_data)
        vis_time_diffrence(model, coords2d, times, temp2d, inverse_scale, animate=True)