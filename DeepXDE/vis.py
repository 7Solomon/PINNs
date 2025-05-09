import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(Loss):
    if isinstance(Loss, dde.model.LossHistory):
        epochs = Loss.steps

        plt.plot(epochs, Loss.loss_train, label='Total Train Loss', color='blue', linestyle='-')
        plt.plot(epochs, Loss.loss_test, label='Total Test Loss', color='orange', linestyle='-')
        #for i, bc_loss in enumerate(Loss.loss_bcs):
        #    plt.plot(epochs, bc_loss, label=f'Train BC {i} Loss', color='blue', linestyle=':')

        #if len(Loss.loss_test_components) > 0:
        #    plt.plot(epochs, Loss.loss_test_components[0], label='Test Residual Loss', color='orange', linestyle='--')
        #    for i in range(1, len(Loss.loss_test_components)): # Start from index 1 for BCs
        #        plt.plot(epochs, Loss.loss_test_components[i], label=f'Test BC {i-1} Loss', color='orange', linestyle=':')


        plt.legend(loc='best') 

    else:
        plt.plot(Loss)
        plt.legend(['Input Data']) 


    plt.grid(True, which="both", ls="--")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss (log Scale)')
    plt.yscale('log')
    plt.show()

def analytical_solution_FLL(x, q=1, L=1, EI=1):
    return (1/12)*x**3 - (1/24) * x**4 - (1/24) * x

def visualize_field(model):
    x = np.linspace(0, 1, 1000)[:, None]
    y = model.predict(x)
    y_analytical = analytical_solution_FLL(x)
    #print("max: ", analytical_solution_FLL(1/2))
    plt.figure()
    plt.plot(x, y, label='predicted', color='red')
    plt.plot(x, y_analytical, label="Analytical Solution", linestyle='--')
    
    plt.plot(x, np.zeros_like(x), label='Balken', color='black', linewidth=3)
    
    
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("SOlu")
    plt.legend()

    #plt.savefig("results/field.png")
    plt.show()

