import numpy as np
import matplotlib.pyplot as plt

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

