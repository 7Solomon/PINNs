import math
from process.mechanic.scale import scale_t, scale_u, scale_x
import numpy as np
import matplotlib.pyplot as plt

def analytical_solution_FLL(x, q=1, L=1, EI=1):
    return (1/12)*x**3 - (1/24) * x**4 - (1/24) * x
def analytical_solution_FES(x, q=1, L=1, EI=1):
    return -(1/2)*q*L*x**2 + (1/6)*q*L*x**3
def analytical_solution_FLL_t(x,t):
    return np.sin(x)*np.cos(4*math.pi*t)
analytical_mapping = {
    'fest_los': analytical_solution_FLL,
    'einspannung': analytical_solution_FES,
    'fest_los_t': analytical_solution_FLL_t,
    #'2D_fest_los': None,
}

def visualize_field_1d(model, type, inverse_scale=None):
    x = np.linspace(0, 1, 1000)[:, None]
    y = model.predict(x)
    y_analytical = analytical_mapping[type](x)
    #print("max: ", analytical_solution_FLL(1/2))
    plt.figure()
    plt.plot(x, -y, label='NEGATIVE predicted', color='red')

    plt.plot(x, y_analytical, label="Analytical Solution", linestyle='--')
    
    plt.plot(x, np.zeros_like(x), label='Balken', color='black', linewidth=3)
    
    
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("SOlu")
    plt.legend()

    #plt.savefig("results/field.png")
    plt.show()

def visualize_field_1d_t(model, type):
    x_comp = np.linspace(0, math.pi, 100) 
    t_comp = np.linspace(0, 1, 100)    
    
    # create mesh
    X_comp_mesh, T_comp_mesh = np.meshgrid(x_comp, t_comp, indexing='xy')
    predict_points = np.hstack((X_comp_mesh.ravel()[:, None], T_comp_mesh.ravel()[:, None]))
    
    # get pred
    Y_pred_scaled_flat = model.predict(predict_points)
    Y_pred_scaled = Y_pred_scaled_flat.reshape(X_comp_mesh.shape)
    y_analytical_scaled = analytical_mapping[type](X_comp_mesh, T_comp_mesh)


    X_plot_mesh = scale_x(X_comp_mesh)
    T_plot_mesh = scale_t(T_comp_mesh)
    
    Y_pred_physical = scale_u(Y_pred_scaled)
    y_analytical_physical = scale_u(y_analytical_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # pred
    contour_pred = axes[0].contourf(X_plot_mesh, T_plot_mesh, Y_pred_physical, levels=50, cmap='viridis')
    fig.colorbar(contour_pred, ax=axes[0], label='Predicted u(x,t)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    axes[0].set_title('pred')

    # analy
    contour_analytical = axes[1].contourf(X_plot_mesh, T_plot_mesh, y_analytical_physical, levels=50, cmap='viridis')
    fig.colorbar(contour_analytical, ax=axes[1], label='Analytical u(x,t)')
    axes[1].set_xlabel('x')
    axes[1].set_title('analytical')
    
    fig.suptitle('FIELD', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



def visualize_field(model, type, inverse_scale=None):
    if type == 'fest_los' or type == 'einspannung':
        visualize_field_1d(model, type, inverse_scale)
    elif type == 'fest_los_t':
       visualize_field_1d_t(model, type)
