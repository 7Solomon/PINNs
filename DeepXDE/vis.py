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
        plt.figure(figsize=(10, 6))
        
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

def plot_mse(mse_values):
    """
    Plot MSE values.
    Args:
        mse_values: List or array of MSE values
    """
    mse_array = np.array(mse_values)
    min_mse = np.min(mse_array)
    max_mse = np.max(mse_array)
    mean_mse = np.mean(mse_array)

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(mse_values)), mse_values)
    plt.xlabel('Epochs [50]')
    plt.ylabel('MSE')
    
    # Two-line title instead of suptitle
    plt.title(f'Mean Squared Error (MSE)\nMin: {min_mse:.2e} | Max: {max_mse:.2e}', 
              fontsize=11)
    
    plt.grid(axis='y', linestyle='--')
    
    return {'mse': plt.gcf()}



def get_deformation_amplifier(u,v):
    all_displacement_magnitudes = np.sqrt(u**2 + v**2)
    max_actual_displacement = np.max(all_displacement_magnitudes)
    
    target_visual_displacement = 0.05 
    
    if max_actual_displacement > 1e-9: 
        amplification = target_visual_displacement / max_actual_displacement
    else:
        amplification = 10
        
    min_amplification = 1.0
    max_amplification = 500.0
    amplification = int(np.clip(amplification, min_amplification, max_amplification))
    return amplification
