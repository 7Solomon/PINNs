import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches

def save_NN():
    # Set up the figure with academic styling
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Define colors
    node_color = '#4a90e2'
    connection_color = '#7c7c7c'
    text_color = '#2c3e50'
    highlight_color = '#e74c3c'
    border_color = '#34495e'
    layer_bg_color = '#ecf0f1'

    # Neural network structure - show first few nodes then dots
    layers = [3, 4, 4, 4, 2]  # nodes to actually draw
    layer_names = ['Input\nLayer', 'Hidden\nLayer 1', '$\\cdots$', 'Hidden\nLayer N', 'Output\nLayer']
    layer_positions = [1.5, 3.5, 5.5, 7.5, 9.5]

    # Store node positions for drawing connections
    node_positions = []

    # Draw layer borders with more prominence and labels inside
    for l, (num_nodes, x_pos, layer_name) in enumerate(zip(layers, layer_positions, layer_names)):
        if layer_name == '$\\cdots$':  # Skip border for dots layer - handle separately
            continue
            
        # Calculate the height needed for the layer (accounting for dots)
        total_height = 5.0  # Fixed height to accommodate dots
        width = 1.4  # Standard width for regular layers
        
        # Create more prominent rounded rectangle border around each layer
        border = FancyBboxPatch(
            (x_pos - width/2, 4 - total_height/2 - 0.3), 
            width, total_height + 0.6,
            boxstyle="round,pad=0.15",
            facecolor=layer_bg_color,
            edgecolor=border_color,
            linewidth=2.5,
            alpha=0.8,
            zorder=1
        )
        ax.add_patch(border)
        
        # Add layer labels INSIDE the border at the bottom
        ax.text(x_pos, 1.5, layer_name, ha='center', va='center', 
            fontsize=11, color=text_color, weight='bold', zorder=5)
        
        # Add layer indices INSIDE the border at the top
        layer_indices = ['$l=0$', '$l=1$', '', '$l=N$', '$l=N+1$']
        if layer_indices[l]:
            ax.text(x_pos, 6.5, layer_indices[l], ha='center', va='center', 
                fontsize=11, color=text_color, style='italic', weight='bold', zorder=5)

    # Add special smaller border for dots layer
    dots_x = layer_positions[2]
    dots_width = 0.6  # Much smaller width
    dots_height = 2.0  # Smaller height too
    dots_border = FancyBboxPatch(
        (dots_x - dots_width/2, 4 - dots_height/2), 
        dots_width, dots_height,
        boxstyle="round,pad=0.1",
        facecolor=layer_bg_color,
        edgecolor=border_color,
        linewidth=2.5,
        alpha=0.8,
        zorder=1
    )
    ax.add_patch(dots_border)

    # Draw connections first (so they appear behind nodes)
    for l in range(len(layers) - 1):
        if layer_names[l] == '$\\cdots$' or layer_names[l+1] == '$\\cdots$':
            continue  # Skip connections involving dots - we'll handle these separately
            
        x1 = layer_positions[l]
        x2 = layer_positions[l + 1]
        
        # Calculate vertical positions for current and next layer
        y1_positions = np.linspace(3.5, 5.5, layers[l])  # visible nodes
        y2_positions = np.linspace(3.5, 5.5, layers[l + 1])  # visible nodes
        
        # Draw all connections between layers
        for i, y1 in enumerate(y1_positions):
            for j, y2 in enumerate(y2_positions):
                ax.plot([x1 + 0.25, x2 - 0.25], [y1, y2], 
                    color=connection_color, alpha=0.4, linewidth=1.0, zorder=2)

    # Draw special connections for the "..." layer
    # From layer 1 to dots layer - connect to multiple dot positions
    x1, x2 = layer_positions[1], layer_positions[2]
    y1_positions = np.linspace(3.5, 5.5, layers[1])
    dot_y_positions = [3.8, 4.0, 4.2]  # Multiple connection points for dots
    
    for y1 in y1_positions:
        for dot_y in dot_y_positions:
            ax.plot([x1 + 0.25, x2 - 0.15], [y1, dot_y], 
                color=connection_color, alpha=0.25, linewidth=0.8, linestyle='--', zorder=2)
    
    # From dots layer to layer N - connect from multiple dot positions
    x1, x2 = layer_positions[2], layer_positions[3]
    y2_positions = np.linspace(3.5, 5.5, layers[3])
    
    for dot_y in dot_y_positions:
        for y2 in y2_positions:
            ax.plot([x1 + 0.15, x2 - 0.25], [dot_y, y2], 
                color=connection_color, alpha=0.25, linewidth=0.8, linestyle='--', zorder=2)

    # Draw nodes and collect positions
    for l, (num_nodes, x_pos, layer_name) in enumerate(zip(layers, layer_positions, layer_names)):
        if layer_name == '$\\cdots$':
            # Draw multiple dots to represent the layer
            ax.text(x_pos, 4.0, '$\\cdots$', ha='center', va='center', 
                fontsize=20, color=text_color, weight='bold', zorder=4)
            continue
            
        # Calculate vertical positions for visible nodes
        visible_nodes = min(num_nodes, 3)  # Show max 3 nodes
        y_positions = np.linspace(3.5, 5.5, visible_nodes)
        
        layer_nodes = []
        for i, y_pos in enumerate(y_positions):
            # Draw node
            circle = Circle((x_pos, y_pos), 0.15, color=node_color, alpha=0.9, zorder=3)
            ax.add_patch(circle)
            
            # Add node labels for mathematical notation
            if l == 0:  # Input layer
                ax.text(x_pos, y_pos, f'$x_{i+1}$', ha='center', va='center', 
                    fontsize=9, color='white', weight='bold', zorder=4)
            elif l == len(layers) - 1:  # Output layer
                ax.text(x_pos, y_pos, f'$y_{i+1}$', ha='center', va='center', 
                    fontsize=9, color='white', weight='bold', zorder=4)
            else:  # Hidden layers
                layer_idx = 'N' if 'Layer N' in layer_name else str(l)
                ax.text(x_pos, y_pos, f'$z^{{({layer_idx})}}_{i+1}$', ha='center', va='center', 
                    fontsize=8, color='white', weight='bold', zorder=4)
            
            layer_nodes.append((x_pos, y_pos))
        
        # Add dots to show there are more nodes with blue circles
        if l != len(layers) - 1:  # Don't add dots to output layer
            ax.text(x_pos, 3.0, '$\\vdots$', ha='center', va='center', 
                fontsize=16, color=node_color, weight='bold', zorder=4)
            
            # Fix the conditional logic for proper LaTeX
            if l == 0:  # Input layer
                bottom_label = '$x_d$'
            else:  # Hidden layers
                layer_idx = 'N' if 'Layer N' in layer_name else str(l)
                bottom_label = f'$z^{{({layer_idx})}}_n$'
            
            # Add blue circle around the bottom label
            bottom_circle = Circle((x_pos, 2.5), 0.15, color=node_color, alpha=0.9, zorder=3)
            ax.add_patch(bottom_circle)
            
            ax.text(x_pos, 2.5, bottom_label, ha='center', va='center', 
                fontsize=8, color='white', weight='bold', zorder=4)
        
        node_positions.append(layer_nodes)

    # Add connection annotations with arrows
    # First layer connection
    ax.annotate('$\\mathbf{W}^{(1)} \\mathbf{x} + \\mathbf{b}^{(1)}$', 
                xy=(2.5, 4.5), xytext=(2.5, 6.0),
                ha='center', va='center', fontsize=12, color=highlight_color, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=highlight_color, linewidth=2),
                arrowprops=dict(arrowstyle='->', color=highlight_color, lw=1.5))

    # General layer connection
    ax.annotate('$\\mathbf{W}^{(l)} \\mathbf{z}^{(l)} + \\mathbf{b}^{(l)}$', 
                xy=(6.5, 4.5), xytext=(6.5, 6.0),
                ha='center', va='center', fontsize=12, color=highlight_color, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=highlight_color, linewidth=2),
                arrowprops=dict(arrowstyle='->', color=highlight_color, lw=1.5))

    # Activation function annotation
    #ax.annotate('$\\sigma(\\cdot)$', 
    #            xy=(8.5, 4.5), xytext=(8.5, 6.0),
    #            ha='center', va='center', fontsize=12, color=highlight_color, weight='bold',
    #            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=highlight_color, linewidth=2),
    #            arrowprops=dict(arrowstyle='->', color=highlight_color, lw=1.5))

    # Add the main equation at the bottom
    equation_text = '$\\mathbf{z}^{(l+1)} = \\sigma(\\mathbf{W}^{(l)} \\mathbf{z}^{(l)} + \\mathbf{b}^{(l)})$'
    ax.text(6, 0.4, equation_text, ha='center', va='center', 
            fontsize=16, color=text_color, weight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.8, edgecolor=text_color))

    # Add title
    ax.text(6, 7.7, 'Basic Neural Network Architecture', ha='center', va='center', 
            fontsize=18, color=text_color, weight='bold')

    # Add parameter annotation
    param_text = '$\\boldsymbol{\\theta} = \\{\\mathbf{W}^{(l)}, \\mathbf{b}^{(l)}\\}_{l=0}^{N}$'
    ax.text(10.5, 0.4, param_text, ha='center', va='center', 
            fontsize=12, color=text_color, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.9, edgecolor=text_color))

    plt.tight_layout()

    # Save the figure in high resolution for paper inclusion
    plt.savefig('neural_network_architecture.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('neural_network_architecture.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    plt.show()
    print("SAVED")



def save_heat_domain():
    """Visualize the 2D heat equation domain with boundary conditions"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Define domain
    x_min, x_max = 0, 0.5
    y_min, y_max = 0, 1.0
    
    # Colors for academic styling
    hot_color = '#e74c3c'  # Red for hot boundary
    cold_color = '#2980b9'  # Blue for cold boundary
    domain_color = '#f8f9fa'  # Very light gray for domain
    text_color = '#2c3e50'
    boundary_color = '#34495e'
    
    # Set limits with some padding
    ax.set_xlim(-0.15, 0.65)
    ax.set_ylim(-0.15, 1.15)
    
    # Draw domain rectangle
    domain_rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           facecolor=domain_color, edgecolor=boundary_color, linewidth=2)
    ax.add_patch(domain_rect)
    
    # Draw boundary conditions with thicker lines
    # Left boundary (T = 100°C) - Hot
    ax.plot([x_min, x_min], [y_min, y_max], color=hot_color, linewidth=8, 
            solid_capstyle='butt', zorder=3)
    
    # Right boundary (T = 0°C) - Cold
    ax.plot([x_max, x_max], [y_min, y_max], color=cold_color, linewidth=8, 
            solid_capstyle='butt', zorder=3)
    
    # Top and bottom boundaries (insulated) - Neumann
    ax.plot([x_min, x_max], [y_max, y_max], color=boundary_color, linewidth=4, 
            linestyle='--', alpha=0.8, zorder=3)
    ax.plot([x_min, x_max], [y_min, y_min], color=boundary_color, linewidth=4, 
            linestyle='--', alpha=0.8, zorder=3)
    
    # Add domain definition in center
    #ax.text(0.25, 0.5, r'$\Omega = [0, 0.5] \times [0, 1]$', 
    #        ha='center', va='center', fontsize=16, color=text_color, weight='bold',
    #        bbox=dict(boxstyle="round,pad=0.4", facecolor='white', edgecolor='lightgray', 
    #                 linewidth=1, alpha=0.95))
    
    # Add boundary condition labels outside domain
    # Left boundary label
    ax.text(-0.08, 0.5, 'T = 100°C', ha='center', va='center', rotation=90,
            fontsize=14, color=hot_color, weight='bold')
    
    # Right boundary label  
    ax.text(0.58, 0.5, 'T = 0°C', ha='center', va='center', rotation=90,
            fontsize=14, color=cold_color, weight='bold')
    
    # Top boundary label
    ax.text(0.25, 1.08, r'$\frac{\partial T}{\partial n} = 0$', ha='center', va='center',
            fontsize=12, color=boundary_color, style='italic')
    
    # Bottom boundary label
    ax.text(0.25, -0.08, r'$\frac{\partial T}{\partial n} = 0$', ha='center', va='center',
            fontsize=12, color=boundary_color, style='italic')
    
    # Add coordinate labels
    ax.text(0.25, -0.12, 'x', ha='center', va='center', fontsize=14, color=text_color)
    ax.text(-0.12, 0.5, 'y', ha='center', va='center', fontsize=14, color=text_color, rotation=90)
    
    # Add corner coordinates
    ax.text(x_min-0.02, y_min-0.02, '(0,0)', ha='right', va='top', fontsize=10, color=text_color)
    ax.text(x_max+0.02, y_max+0.02, '(0.5,1)', ha='left', va='bottom', fontsize=10, color=text_color)
    
    # Add governing equation as title
    #ax.text(0.25, 1.25, r'$\frac{\partial T}{\partial t} = \alpha \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} \right)$', 
    #        ha='center', va='center', fontsize=18, color=text_color, weight='bold')
    
    # Add initial condition
    ax.text(0.25, 1.25, r'Initial condition: $T(x,y,0) = 0$°C', 
            ha='center', va='center', fontsize=12, color=text_color, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Set aspect ratio
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('heat_domain.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('heat_domain.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    plt.show()
    print("Heat domain SAVED")

if __name__ == "__main__":
    save_heat_domain()