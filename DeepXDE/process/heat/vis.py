import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation


def visualize_steady_field(model, inverse_scale=None):
    # Create grid
    nx, ny = 100, 50
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    points = np.vstack((X.flatten(), Y.flatten())).T
    predictions = model.predict(points)
    
    if inverse_scale:
        predictions = inverse_scale(predictions)
    
    Z = predictions.reshape(ny, nx)
    
    #PLOT
    plt.figure(figsize=(10, 5))
    plt.contourf(X, Y, Z, 50, cmap=cm.jet)
    plt.colorbar(label='Temperature')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Predicted Heat Distribution')
    plt.tight_layout()
    #plt.savefig('heat_field.png', dpi=300)
    plt.show()
    
    #return Z
def visualize_time_dependent_field(model, inverse_scale=None, 
                                   animate=True, save_animation=False):
    
    # Create spatial grid
    nx, ny, nt = 100, 50, 100
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 1, ny)
    t = np.linspace(0, 100, nt)
    X, Y, T = np.meshgrid(x, y, t)
    points = np.vstack((X.flatten(), Y.flatten(), T.flatten())).T
    
    pred = model.predict(points)
    if inverse_scale:
        pred = inverse_scale(pred)
    pred = pred.reshape(ny, nx, nt)

    if animate:
        fig, ax = plt.subplots(figsize=(10, 5))
        #plt.close()
        
        # First frame
        cont = ax.contourf(x, y, pred[:,:,0], 50, cmap=cm.jet, vmin=0.0, vmax=100.0)
        cbar = fig.colorbar(cont, ax=ax)
        cbar.set_label('Temperature')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Heat Distribution[t={t[0]:.3f}]')
        
        def update(frame):
            ax.clear()
            cont = ax.contourf(x, y, pred[:,:,frame], 50, cmap=cm.jet, vmin=0.0, vmax=100.0)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'Heat Distribution[t={t[frame]:.3f}]')
            return cont
        
        ani = animation.FuncAnimation(fig, update, frames=nt, interval=100)
        
        if save_animation:
            ani.save('heat_evolution.mp4', writer='ffmpeg', dpi=300)
            
        plt.tight_layout()
        plt.show()
        
        #return ani
    
    #else:
    #    # Create subplot grid
    #    rows = int(np.ceil(np.sqrt(nt)))
    #    cols = int(np.ceil(nt / rows))
    #    
    #    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    #    axes = axes.flatten()
    #    
    #    for i in range(nt):
    #        if i < len(t_points):
    #            cont = axes[i].contourf(X, Y, results[i], 50, cmap=cm.jet, vmin=vmin, vmax=vmax)
    #            axes[i].set_xlabel('x')
    #            axes[i].set_ylabel('y')
    #            axes[i].set_title(f't = {t_points[i]:.3f}')
    #        else:
    #            axes[i].axis('off')
    #    
    #    # colorbar
    #    fig.subplots_adjust(right=0.8)
    #    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    #    cbar = fig.colorbar(cont, cax=cbar_ax)
    #    cbar.set_label('Temperature')
    #    
    #    plt.tight_layout()
    #    plt.savefig('heat_snapshots.png', dpi=300, bbox_inches='tight')
    #    plt.show()
    #    
    #    #return results

def visualize_field(model, type, inverse_scale=None):
    if type == 'steady':
        visualize_steady_field(model, inverse_scale=inverse_scale)
    elif type == 'transient':
        visualize_time_dependent_field(model,inverse_scale=inverse_scale, animate=True)
