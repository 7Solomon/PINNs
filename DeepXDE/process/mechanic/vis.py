import math
from utils.function_utils import is_inside_polygon
from process.mechanic.scale import  *
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon


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



def visualize_cooks_membrane(model, resolution=20, scale_factor=1.0):

    domain_points = np.array([
        [0, 0],
        [48, 44],
        [48, 60],
        [0, 44],
    ])
    
    # Create a grid of points within the domain
    x_min, y_min = np.min(domain_points, axis=0)
    x_max, y_max = np.max(domain_points, axis=0)
    
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create points array for prediction
    points = np.vstack((X.flatten(), Y.flatten())).T
    
    mask = is_inside_polygon(points, domain_points)
    points_inside = points[mask]
    
    # Predict displacements
    if points_inside.size > 0:
        predictions = model.predict(scale_x(points_inside))
        predictions = rescale_x
        u_x = predictions[:, 0].reshape(-1)  # x-displacement
        u_y = predictions[:, 1].reshape(-1)  # y-displacement
        
        # Calculate deformed points
        deformed_points = points_inside.copy()
        deformed_points[:, 0] += u_x * scale_factor
        deformed_points[:, 1] += u_y * scale_factor
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot original domain
        plt.subplot(1, 3, 1)
        plt.title("Original Domain")
        plt.gca().add_patch(Polygon(domain_points, fill=False, edgecolor='black'))
        plt.scatter(points_inside[:, 0], points_inside[:, 1], c='blue', s=10, alpha=0.5)
        plt.axis('equal')
        plt.grid(True)
        
        # Plot deformed domain
        plt.subplot(1, 3, 2)
        plt.title(f"Deformed Domain (scale={scale_factor})")
        plt.scatter(deformed_points[:, 0], deformed_points[:, 1], c='red', s=10, alpha=0.5)
        plt.axis('equal')
        plt.grid(True)
        
        # Plot displacement magnitude
        plt.subplot(1, 3, 3)
        plt.title("Displacement Magnitude")
        displacement_mag = np.sqrt(u_x**2 + u_y**2)
        sc = plt.scatter(points_inside[:, 0], points_inside[:, 1], 
                        c=displacement_mag, cmap='jet', s=30, alpha=0.8)
        plt.colorbar(sc, label="Displacement magnitude")
        plt.gca().add_patch(Polygon(domain_points, fill=False, edgecolor='black'))
        plt.axis('equal')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Also visualize displacement vectors
        plt.figure(figsize=(10, 8))
        plt.title("Displacement Field")
        plt.quiver(points_inside[:, 0], points_inside[:, 1], 
                u_x, u_y, angles='xy', scale_units='xy', 
                scale=0.1/scale_factor, color='red')
        plt.gca().add_patch(Polygon(domain_points, fill=False, edgecolor='black'))
        plt.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No points inside the domain were found.")

def visualize_field(model, type, inverse_scale=None):
    if type == 'fest_los' or type == 'einspannung':
        visualize_field_1d(model, type, inverse_scale)
    elif type == 'fest_los_t':
       visualize_field_1d_t(model, type)
    elif type == 'cooks':
        visualize_cooks_membrane(model)