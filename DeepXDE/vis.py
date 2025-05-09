import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(Loss):
    if isinstance(Loss, dde.model.LossHistory):
        epochs = Loss.steps        
        labels =['PDE', 'DBC links', 'DBC rechts', 'NBC links', 'NBC rechts', 'Data/Other']
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
        if Loss.loss_test and len(Loss.loss_test[0]) > 0:
            loss_test_np = np.array(Loss.loss_test)
            num_test_components = loss_test_np.shape[1]
            for i in range(num_test_components):
                component_label = labels[i] if i < len(labels) else f'Test Comp {i+1}'
                plt.plot(epochs, loss_test_np[:, i], label=f'{component_label} (Test)', linestyle='--')
        else:
            print('No testing loss')
        plt.legend(loc='best')
    else:
        plt.legend(loc='best')
        plt.plot(Loss, label='Loss')


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

