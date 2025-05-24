import math
from utils.metadata import Domain
from process.mechanic.scale import  *
import numpy as np
import matplotlib.pyplot as plt
from vis import get_2d_domain



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

def visualize_field_1d(model, type):
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

def visualize_field_2d(model, type):
    domain_vars = Domain(
        spatial={
            'x':(0,10),
            'y':(0,1)
        }
    )
    domain = get_2d_domain(domain_vars, scale_x, scale_y)
    points, X, Y, nx, ny = domain['normal']
    scaled_points, scaled_X, scaled_Y, nx, ny = domain['scaled']

    predictions = model.predict(scaled_points)
    
    # Debug info
    print(f"Domain: x=(0,10), y=(0,1) - should be horizontal beam")
    print(f"Points shape: {scaled_points.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Grid dimensions: nx={nx}, ny={ny}")
    
    # Flatten coordinate arrays
    if len(scaled_X.shape) > 1:
        scaled_X_flat = scaled_X.flatten()
        scaled_Y_flat = scaled_Y.flatten()
    else:
        scaled_X_flat = scaled_X
        scaled_Y_flat = scaled_Y
    
    # Get actual number of points
    total_points = predictions.shape[0]
    
    # Subsample for visualization (25M points is too many)
    if total_points > 100000:  # If more than 100k points, subsample
        step = total_points // 50000  # Take every nth point to get ~50k points
        indices = np.arange(0, total_points, step)
        scaled_X_flat = scaled_X_flat[indices]
        scaled_Y_flat = scaled_Y_flat[indices]
        predictions = predictions[indices]
        total_points = len(indices)
        print(f"Subsampled to {total_points} points for visualization")
    
    # Ensure coordinate arrays match prediction length
    if len(scaled_X_flat) != total_points:
        scaled_X_flat = scaled_X_flat[:total_points]
        scaled_Y_flat = scaled_Y_flat[:total_points]
    
    # Convert back to original domain coordinates for proper visualization
    x_min, x_max = domain_vars.spatial['x']
    y_min, y_max = domain_vars.spatial['y']
    
    # Assuming scaled coordinates are in [0,1], map back to original domain
    original_X = scaled_X_flat * (x_max - x_min) + x_min
    original_Y = scaled_Y_flat * (y_max - y_min) + y_min
    
    print(f"Original X range: {original_X.min():.3f} to {original_X.max():.3f}")
    print(f"Original Y range: {original_Y.min():.3f} to {original_Y.max():.3f}")
    
    # Infer reasonable grid dimensions
    sqrt_points = int(np.sqrt(total_points))
    if sqrt_points * sqrt_points == total_points:
        nx = ny = sqrt_points
    else:
        # Find best rectangular grid that matches beam aspect ratio
        aspect_ratio = (x_max - x_min) / (y_max - y_min)  # Should be 10
        ny = int(np.sqrt(total_points / aspect_ratio))
        nx = total_points // ny
    
    print(f"Using grid dimensions: nx={nx}, ny={ny}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Displacement magnitude
    displacement_magnitude = np.sqrt(predictions[:, 0]**2 + predictions[:, 1]**2)
    
    try:
        if nx * ny == total_points:
            displacement_magnitude_2d = displacement_magnitude.reshape(ny, nx)
            X_reshaped = original_X.reshape(ny, nx)
            Y_reshaped = original_Y.reshape(ny, nx)
            
            contour = axes[0, 0].contourf(X_reshaped, Y_reshaped, 
                                          displacement_magnitude_2d, levels=20, cmap='viridis')
            axes[0, 0].set_title("Displacement Magnitude")
            plt.colorbar(contour, ax=axes[0, 0])
        else:
            raise ValueError("Grid dimensions don't match")
            
    except ValueError:
        # Fall back to scatter plot
        scatter = axes[0, 0].scatter(original_X, original_Y, c=displacement_magnitude, 
                                   cmap='viridis', s=1, alpha=0.7)
        axes[0, 0].set_title("Displacement Magnitude (scatter)")
        plt.colorbar(scatter, ax=axes[0, 0])
    
    # 2. Deformed shape
    scale_factor = 5.0  # Exaggerate deformation for visibility
    deformed_X = original_X + scale_factor * predictions[:, 0]
    deformed_Y = original_Y + scale_factor * predictions[:, 1]
    
    axes[0, 1].scatter(original_X, original_Y, c='blue', s=0.5, alpha=0.3, label='Original')
    axes[0, 1].scatter(deformed_X, deformed_Y, c='red', s=0.5, alpha=0.7, label=f'Deformed (×{scale_factor})')
    axes[0, 1].set_title("Deformed Shape")
    axes[0, 1].legend()
    axes[0, 1].set_aspect('equal')
    
    # 3. X-displacement
    scatter_x = axes[1, 0].scatter(original_X, original_Y, c=predictions[:, 0], 
                                 cmap='RdBu_r', s=1, alpha=0.7)
    axes[1, 0].set_title("X-Displacement")
    plt.colorbar(scatter_x, ax=axes[1, 0])
    
    # 4. Y-displacement
    scatter_y = axes[1, 1].scatter(original_X, original_Y, c=predictions[:, 1], 
                                 cmap='RdBu_r', s=1, alpha=0.7)
    axes[1, 1].set_title("Y-Displacement")
    plt.colorbar(scatter_y, ax=axes[1, 1])
    
    # Add beam outline to all plots
    beam_outline = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]])
    
    for ax in axes.flat:
        ax.plot(beam_outline[:, 0], beam_outline[:, 1], 'k-', linewidth=2, alpha=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(x_min-0.5, x_max+0.5)
        ax.set_ylim(y_min-0.5, y_max+0.5)
    
    plt.tight_layout()
    #plt.savefig("Field_2d_comprehensive.png", dpi=300, bbox_inches='tight')
    #plt.show()^
    return {'field': fig}
