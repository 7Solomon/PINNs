import math
from utils.metadata import Domain
from process.mechanic.scale import  *
import numpy as np
import matplotlib.pyplot as plt
from vis import get_2d_domain

from domain_vars import fest_lost_2d_domain



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

def visualize_field_1d(model, **kwargs):
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
    return {'field': plt.gcf()}
    #plt.savefig("Field_1d.png")
    #plt.show()


def visualize_field_2d(model, **kwargs):
    """
    Visualizes the 2D displacement field predicted by a model.

    Args:
        model: The trained model with a .predict() method.
        domain_vars: A Domain object or similar structure defining spatial bounds.
        scale_x: Scaling factor for x-coordinates.
        scale_y: Scaling factor for y-coordinates.

    Returns:
        A dictionary containing the matplotlib Figure object.
    """

    # Get domain and points
    domain = get_2d_domain(fest_lost_2d_domain, scale_x, scale_y)
    points, X, Y, nx, ny = domain['normal']
    scaled_points, scaled_X, scaled_Y, _, _ = domain['scaled']

    # Get predictions
    predictions = model.predict(scaled_points)
    # predictions = scale_u(predictions) # Uncomment if you need to scale output

    # --- Extract min/max for bounds (Needed for outline/limits) ---
    x_min, x_max = fest_lost_2d_domain.spatial['x']
    y_min, y_max = fest_lost_2d_domain.spatial['y']
    # -------------------------------------------------------------

    # Create visualization (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('2D Field Visualization', fontsize=16) # Add an overall title

    # Calculate displacement magnitude (assuming X, Y are now 2D: [ny, nx])
    displacement_magnitude = np.sqrt(predictions[:, 0]**2 + predictions[:, 1]**2)
    displacement_magnitude_2d = displacement_magnitude.reshape(ny, nx)

    # 1. Displacement Magnitude (Top-Left)
    ax = axes[0, 0]
    contour = ax.contourf(X, Y, displacement_magnitude_2d, levels=20, cmap='viridis')
    ax.set_title("Displacement Magnitude")
    plt.colorbar(contour, ax=ax, shrink=0.8) # Add colorbar
    ax.set_aspect('equal') # Often good for physical fields

    # Calculate deformed shape (assuming X, Y are 2D: [ny, nx])
    u_x = predictions[:, 0].reshape(ny, nx)
    u_y = predictions[:, 1].reshape(ny, nx)
    scale_factor = 5.0  # Adjust for visibility
    deformed_X = X + scale_factor * u_x
    deformed_Y = Y + scale_factor * u_y

    # 2. Deformed Shape (Top-Right)
    ax = axes[0, 1]
    ax.scatter(X, Y, c='blue', s=0.5, alpha=0.3, label='Original')
    ax.scatter(deformed_X, deformed_Y, c='red', s=0.5, alpha=0.7, label=f'Deformed (×{scale_factor})')
    ax.set_title("Deformed Shape")
    ax.legend()
    ax.set_aspect('equal') # Crucial for deformation plots

    # 3. X-Displacement (Bottom-Left)
    ax = axes[1, 0]
    # Use reshape on the color data if X, Y are 2D, or plot flattened X, Y
    scatter_x = ax.scatter(X, Y, c=predictions[:, 0], cmap='RdBu_r', s=1, alpha=0.7)
    ax.set_title("X-Displacement ($u_x$)")
    plt.colorbar(scatter_x, ax=ax, shrink=0.8)
    ax.set_aspect('equal')

    # 4. Y-Displacement (Bottom-Right)
    ax = axes[1, 1]
    scatter_y = ax.scatter(X, Y, c=predictions[:, 1], cmap='RdBu_r', s=1, alpha=0.7)
    ax.set_title("Y-Displacement ($u_y$)")
    plt.colorbar(scatter_y, ax=ax, shrink=0.8)
    ax.set_aspect('equal')

    # --- Add beam outline and format all plots ---
    beam_outline = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]])
    
    for ax in axes.flat:
        ax.plot(beam_outline[:, 0], beam_outline[:, 1], 'k-', linewidth=1.5, alpha=0.9, label='Boundary')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Y-coordinate')
        # Set limits with a small buffer
        ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
        ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    # Add legend only once if using the 'Boundary' label
    axes[0,1].legend() # Re-call legend on one plot to include 'Boundary' if needed

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    
    # plt.savefig("Field_2d_comprehensive.png", dpi=300, bbox_inches='tight')
    # plt.show()
    
    return {'field': fig}
