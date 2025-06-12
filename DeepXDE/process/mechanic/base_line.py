import math
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

def analytical_solution_FLL(x, q=1, L=1, EI=1):
    return (1/12)*x**3 - (1/24) * x**4 - (1/24) * x
def analytical_solution_FES(x, q=1, L=1, EI=1):
    return -(1/2)*q*L*x**2 + (1/6)*q*L*x**3
def analytical_solution_FLL_t(x,t):
    return np.sin(x)*np.cos(4*math.pi*t)
#def solution_2d_einspannung(X, Y):
#    """
#    Interpolated solution for 2D einspannung from COMSOL data
#    Returns displacement components [u_x, u_y] for given coordinates
#    """
#    data = load_comsol_data('/home/imb-user/Documents/johannes/PINNs/DeepXDE/BASELINE/mechanic/einspannung_2d.txt')
#
#    X_comsol = data[:, 0]
#    Y_comsol = data[:, 1]
#    u_comsol = data[:, 2]
#    v_comsol = data[:, 3]
#    
#    # Flatten input coordinates if they're meshgrid arrays
#    X_flat = X.flatten() if hasattr(X, 'flatten') else np.array(X)
#    Y_flat = Y.flatten() if hasattr(Y, 'flatten') else np.array(Y)
#    
#    # Interpolate u and v components
#    u_interp = griddata((X_comsol, Y_comsol), u_comsol, (X_flat, Y_flat), method='linear', fill_value=0.0)
#    v_interp = griddata((X_comsol, Y_comsol), v_comsol, (X_flat, Y_flat), method='linear', fill_value=0.0)
#    
#    # Reshape back if original input was 2D
#    if hasattr(X, 'shape') and len(X.shape) > 1:
#        u_interp = u_interp.reshape(X.shape)
#        v_interp = v_interp.reshape(X.shape)
#    
#    return np.column_stack([u_interp, v_interp]) if len(u_interp.shape) == 1 else np.stack([u_interp, v_interp], axis=-1)

base_mapping = {
    'fest_los': analytical_solution_FLL,
    'einspannung': analytical_solution_FES,
    'fest_los_t': analytical_solution_FLL_t,
    #'einspannung_2d': solution_2d_einspannung
    }
