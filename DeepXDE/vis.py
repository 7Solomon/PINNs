from utils.metadata import Domain 
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch
import mplcursors

from matplotlib import cm
import matplotlib.animation as animation


def plot_loss(Loss, labels=None):
    """
    Plot loss history with customizable labels.
    
    Args:
        Loss: LossHistory object or list/array of loss values
        labels: List of labels for loss components. If None, uses generic labels.
    """
    
    if isinstance(Loss, dde.model.LossHistory):
        epochs = Loss.steps
        
        if Loss.loss_train and len(Loss.loss_train[0]) > 0:
            loss_train_np = np.array(Loss.loss_train)
            num_components = loss_train_np.shape[1]
            
            if labels is None:
                labels = [f'Component {i+1}' for i in range(num_components)]
            elif len(labels) < num_components:
                labels = list(labels) + [f'Component {i+1}' for i in range(len(labels), num_components)]
            

            # Here PLOT LOSS
            for i in range(num_components):
                plt.plot(epochs, loss_train_np[:, i], label=labels[i])
        else:
            print('No training loss data available')
            
        plt.legend(loc='best')
    else:
        # Handle case where Loss is just a simple array/list
        plt.plot(Loss, label='Total Loss')
        plt.legend()

    plt.grid(True, which="both", ls="--")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss (Log Scale)')
    plt.yscale('log')
    
    return {'loss': plt.gcf()}

        

def get_2d_domain(domain_variabeles: Domain, scale):
    min_x, max_x = list(domain_variabeles.spatial.values())[0]
    min_y, max_y = list(domain_variabeles.spatial.values())[1]
    
    # Create grid
    nx, ny = 100, 50
    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    X, Y = np.meshgrid(x, y)

    scaled_X = X.copy() / getattr(scale, 'Lx', getattr(scale, 'L', 1))
    scaled_Y = Y.copy() / getattr(scale, 'Ly', getattr(scale, 'L', 1))

    points = np.vstack((X.ravel(), Y.ravel())).T
    scaled_points = np.vstack((scaled_X.ravel(), scaled_Y.ravel())).T  

    return {'normal':[points, X, Y, nx, ny], 'scaled': [scaled_points, scaled_X, scaled_Y, nx, ny]}

def get_2d_time_domain(domain_variabeles:Domain, scale):
    min_x, max_x = list(domain_variabeles.spatial.values())[0]
    min_y, max_y = list(domain_variabeles.spatial.values())[1]
    min_t, max_t = list(domain_variabeles.temporal.values())[0]


    # Create grid
    nx, ny, nt = 100, 50, 100
    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    t = np.linspace(min_t, max_t, nt)
    X, Y, T = np.meshgrid(x, y, t)

    scaled_X = X.copy() / getattr(scale, 'Lx', getattr(scale, 'L', 1))
    scaled_Y = Y.copy() / getattr(scale, 'Ly', getattr(scale, 'L', 1))
    scaled_T = T.copy() / getattr(scale, 'T', getattr(scale, 't', 1))

    points = np.vstack((X.flatten(), Y.flatten(), T.flatten())).T
    scaled_points = np.vstack((scaled_X.flatten(), scaled_Y.flatten(), scaled_T.flatten())).T

    return {'normal':[points, X, Y, T, nx, ny, nt], 'scaled': [scaled_points, scaled_X, scaled_Y, scaled_T,  nx, ny, nt]}

