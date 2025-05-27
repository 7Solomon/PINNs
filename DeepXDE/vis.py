from utils.metadata import Domain 
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch
import mplcursors

from matplotlib import cm
import matplotlib.animation as animation


def plot_loss(Loss):
    if isinstance(Loss, dde.model.LossHistory):
        epochs = Loss.steps
        labels =['PDE', 'links', 'rechts', 'initial']
        # training losses
        if Loss.loss_train and len(Loss.loss_train[0]) > 0:
            loss_train_np = np.array(Loss.loss_train)
            num_train_components = loss_train_np.shape[1]
            for i in range(num_train_components):
                component_label = labels[i] if i < len(labels) else f'Train Comp {i+1}'
                plt.plot(epochs, loss_train_np[:, i], label=f'{component_label}')
        else:
            print('No training loss')

        # testing losses
        #if Loss.loss_test and len(Loss.loss_test[0]) > 0:
        #    loss_test_np = np.array(Loss.loss_test)
        #    num_test_components = loss_test_np.shape[1]
        #    for i in range(num_test_components):
        #        component_label = labels[i] if i < len(labels) else f'Test Comp {i+1}'
        #        plt.plot(epochs, loss_test_np[:, i], label=f'{component_label} (Test)', linestyle='--')
        #else:
        #    print('No testing loss')
        plt.legend(loc='best')
    else:
        #plt.legend(loc='best')
        plt.plot(Loss, label='Loss')


    plt.grid(True, which="both", ls="--")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss (log Scale)')
    plt.yscale('log')
    #plt.savefig('loss.png', dpi=300)
    #plt.show()
    return {'loss': plt.gcf()}

        

def get_2d_domain(domain_variabeles: Domain, scale_x, scale_y):
    min_x, max_x = list(domain_variabeles.spatial.values())[0]
    min_y, max_y = list(domain_variabeles.spatial.values())[1]
    
    # Create grid
    nx, ny = 100, 50
    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    X, Y = np.meshgrid(x, y)

    scaled_X = scale_x(X)
    scaled_Y = scale_y(Y)
    
    points = np.vstack((X.ravel(), Y.ravel())).T
    scaled_points = np.vstack((scaled_X.ravel(), scaled_Y.ravel())).T  

    return {'normal':[points, X, Y, nx, ny], 'scaled': [scaled_points, scaled_X, scaled_Y, nx, ny]}

def get_2d_time_domain(domain_variabeles:Domain, scale_x, scale_y, scale_t):
    min_x, max_x = list(domain_variabeles.spatial.values())[0]
    min_y, max_y = list(domain_variabeles.spatial.values())[1]
    min_t, max_t = list(domain_variabeles.temporal.values())[0]


    # Create grid
    nx, ny, nt = 100, 50, 100
    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    t = np.linspace(min_t, max_t, nt)
    X, Y, T = np.meshgrid(x, y, t)

    scaled_X = scale_x(X)
    scaled_Y = scale_y(Y)
    scaled_T = scale_t(T)
    

    
    points = np.vstack((X.flatten(), Y.flatten(), T.flatten())).T
    scaled_points = np.vstack((scaled_X.flatten(), scaled_Y.flatten(), scaled_T.flatten())).T  

    return {'normal':[points, X, Y, T, nx, ny, nt], 'scaled': [scaled_points, scaled_X, scaled_Y, scaled_T,  nx, ny, nt]}
    
