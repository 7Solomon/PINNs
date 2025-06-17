import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np


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

