import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch
import mplcursors

from matplotlib import cm
import matplotlib.animation as animation


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


def visualize_time_dependent_field(model, domain_variabeles, inverse_scale=None, 
                                   animate=True, save_animation=False):
    x_min, x_max = domain_variabeles['x_min'], domain_variabeles['x_max']
    y_min, y_max = domain_variabeles['y_min'], domain_variabeles['y_max']
    t_min, t_max = domain_variabeles['t_min'], domain_variabeles['t_max']
    min_val, max_val = domain_variabeles['min_val'], domain_variabeles['max_val']

    # Create spatial grid
    nx, ny, nt = 100, 50, 100
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    t = np.linspace(t_min, t_max, nt)
    X, Y, T = np.meshgrid(x, y, t)
    points = np.vstack((X.flatten(), Y.flatten(), T.flatten())).T
    
    pred = model.predict(points)
    if inverse_scale:
        pred = inverse_scale(pred)
    pred = pred.reshape(ny, nx, nt)

    if animate:
        fig, ax = plt.subplots(figsize=(5*x_max, 5*y_max))
        #plt.close()
        
        # First frame
        cont = ax.contourf(x, y, pred[:,:,0], 50, cmap=cm.jet, vmin=min_val, vmax=max_val)
        cbar = fig.colorbar(cont, ax=ax)
        cbar.set_label('Field Prediction')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Distribution at [t={t[0]:.3f}]')
        
        def update(frame):
            ax.clear()
            cont = ax.contourf(x, y, pred[:,:,frame], 50, cmap=cm.jet, vmin=min_val, vmax=max_val)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'Distribution[t={t[frame]:.3f}]')
            cbar.update_normal(cont)
            return cont
        
        ani = animation.FuncAnimation(fig, update, frames=nt, interval=100)
        
        #if save_animation:
        #    ani.save('ka_volution.mp4', writer='ffmpeg', dpi=300)
            
        plt.tight_layout()
        plt.show()
    else:
        raise NotImplementedError('DAS NIX GUT!')
        

def visualize_steady_field(model, domain_variabels, inverse_scale=None, ground=None):

    min_x, max_x = domain_variabels['x_min'], domain_variabels['x_max']
    min_y, max_y = domain_variabels['y_min'], domain_variabels['y_max']
    # Create grid
    nx, ny = 100, 50
    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    X, Y = np.meshgrid(x, y)
    
    points = np.vstack((X.flatten(), Y.flatten())).T
    predictions = model.predict(points)
    
    if inverse_scale:
        predictions = inverse_scale(predictions)
    
    Z = predictions.reshape(ny, nx)

    if ground != None:
        Z = Z - ground

    #PLOT
    plt.figure(figsize=(5*max_x, 5*max_y))
    plt.contourf(X, Y, Z, 50, cmap=cm.jet)
    plt.colorbar(label='Steady Field')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Predicted Distribution')
    plt.tight_layout()
    #plt.savefig('heat_field.png', dpi=300)
    plt.show()

def vis_steady_diffrence(model, domain, temp, inverse_scale=None):

    predictions_np = model.predict(domain) 

    if inverse_scale:
        predictions_np = inverse_scale(predictions_np)
    
    # Convert temp tensor to numpy array
    temp_np = temp.numpy()
    predictions_flat = predictions_np.squeeze()
    temp_flat = temp_np.squeeze()
    
    # Calculate the difference
    div = predictions_flat - temp_flat
   
    # Get domain coordinates as numpy array for plotting
    domain_np = domain.numpy() # Shape (N, 2)
    x_coords = domain_np[:, 0]
    y_coords = domain_np[:, 1]
    min_x_val, max_x_val = np.min(x_coords), np.max(x_coords)
    min_y_val, max_y_val = np.min(y_coords), np.max(y_coords)
    
    delta_x = max_x_val - min_x_val
    delta_y = max_y_val - min_y_val

    fig_width = 8
    if delta_x > 1e-6:
        aspect_ratio = delta_y / delta_x
        fig_height = fig_width * aspect_ratio
    else: 
        fig_height = fig_width
    
    fig_height = max(3.0, min(fig_height, 10.0))

    #PLOT
    plt.figure(figsize=(fig_width, fig_height))
    
    try:
        contour_plot = plt.tricontourf(x_coords, y_coords, div, 50, cmap=cm.jet)
        plt.colorbar(contour_plot, label='Difference (Prediction - Ground Truth)')
    except RuntimeError as e:
        print(f"Warning: tricontourf failed with error: {e}. Falling back to scatter plot.")
        scatter_plot = plt.scatter(x_coords, y_coords, c=div, cmap=cm.jet, s=15) 
        plt.colorbar(scatter_plot, label='Difference (Prediction - Ground Truth)')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Difference between Prediction and Ground Truth')
    plt.tight_layout()
    plt.show()

def vis_time_diffrence(
    model,
    coords2d,     # Tensor [N,2]
    times,        # Tensor [M]
    temp2d,       # Tensor [N,M]
    inverse_scale=None,
    animate=True,
    num_frames=10,
):
    # 1) flatten spatial & time into a single (N·M,3) array
    coords = coords2d.numpy()        # (N,2)
    ts = times.numpy()               # (M,)
    tt = temp2d.numpy()              # (N,M)

    N, M = coords.shape[0], ts.shape[0]
    coords_rep = np.repeat(coords, M, axis=0)      # (N·M,2)
    times_rep  = np.tile(ts, N)                   # (N·M,)
    domain_np  = np.column_stack((coords_rep, times_rep))  # (N·M,3)
    temp_np    = tt.flatten()                     # (N·M,)

    # 2) get model predictions
    preds = model.predict(domain_np)              # (N·M,1) or (N·M,)
    if inverse_scale:
        preds = inverse_scale(preds)
    pred_np = preds.squeeze()

    # 3) now exactly as your old code but using domain_np[:,2] for time:
    unique_times = np.unique(times_rep)
    time_indices = np.argsort(unique_times)
    
    if len(unique_times) > num_frames and not animate:
        frame_indices = np.linspace(0, len(unique_times)-1, num_frames, dtype=int)
        selected_times = unique_times[time_indices[frame_indices]]
    else:
        selected_times = unique_times[time_indices]
    
    if animate:

        plots = []
        min_diff = float('inf')
        max_diff = float('-inf')
        
        # First precompute all frames and find global min/max for colorbar
        for t_val in selected_times:
            # Get points at this time
            time_mask = np.isclose(domain_np[:, 2], t_val)
            x_t = domain_np[time_mask, 0]
            y_t = domain_np[time_mask, 1]
            pred_t = pred_np[time_mask].squeeze()
            temp_t = temp_np[time_mask].squeeze()
            
            # Calculate difference
            diff_t = pred_t - temp_t
            
            # Update global min/max
            min_diff = min(min_diff, np.min(diff_t))
            max_diff = max(max_diff, np.max(diff_t))
            
            plots.append((x_t, y_t, diff_t, t_val))
        
        # Create figure + initial contour with diverging cmap centered at zero
        fig, ax = plt.subplots(figsize=(10, 6))
        x_0, y_0, diff_0, t_0 = plots[0]
        cont = ax.tricontourf(
            x_0, y_0, diff_0, 50,
            cmap="seismic",         # <-- diverging
            vmin=-max(abs(min_diff), abs(max_diff)),
            vmax= max(abs(min_diff), abs(max_diff)),
        )
        # create colorbar once
        cbar = fig.colorbar(cont, ax=ax, pad=0.02)
        cbar.set_label('Difference (Prediction - Ground Truth)', fontsize=12)
        cbar.ax.tick_params(labelsize=11)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        title = ax.set_title(f'Difference at t={t_0:.3f}')

        def update_frame(i):
            ax.clear()
            x_i, y_i, diff_i, t_i = plots[i]
            cont = ax.tricontourf(
                x_i, y_i, diff_i, 50,
                cmap="seismic",
                vmin=-max(abs(min_diff), abs(max_diff)),
                vmax= max(abs(min_diff), abs(max_diff)),
            )
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'Difference at t={t_i:.3f}')
            # just update the existing colorbar scale
            cbar.update_normal(cont)
            return cont

        ani = animation.FuncAnimation(
            fig, update_frame,
            frames=len(plots), interval=200
        )
        plt.tight_layout()
        plt.show()
        
        # Uncomment to save animation
        # ani.save('time_difference.mp4', writer='ffmpeg', dpi=300)
        
    else:
        # Create a grid of plots for selected time points
        num_cols = min(4, len(selected_times))
        num_rows = (len(selected_times) + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*4, num_rows*3))
        if num_rows == 1 and num_cols == 1:
            axes = np.array([axes])  # Make it indexable when only one plot
        axes = axes.flatten()
        
        # Find global min/max for consistent colorbar
        min_diff = float('inf')
        max_diff = float('-inf')
        
        for i, t_val in enumerate(selected_times):
            # Get points at this time
            time_mask = np.isclose(domain_np[:, 2], t_val)
            x_t = domain_np[time_mask, 0]
            y_t = domain_np[time_mask, 1]
            pred_t = pred_np[time_mask].squeeze()
            temp_t = temp_np[time_mask].squeeze()
            
            # Calculate difference
            diff_t = pred_t - temp_t
            
            # Update global min/max
            min_diff = min(min_diff, np.min(diff_t))
            max_diff = max(max_diff, np.max(diff_t))
            
        # Now create each subplot with consistent color range
        for i, t_val in enumerate(selected_times):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get points at this time
            time_mask = np.isclose(domain_np[:, 2], t_val)
            x_t = domain_np[time_mask, 0]
            y_t = domain_np[time_mask, 1]
            pred_t = pred_np[time_mask].squeeze()
            temp_t = temp_np[time_mask].squeeze()
            
            # Calculate difference
            diff_t = pred_t - temp_t
            
            try:
                cont = ax.tricontourf(x_t, y_t, diff_t, 50, cmap=cm.jet, 
                                      vmin=min_diff, vmax=max_diff)
            except RuntimeError as e:
                print(f"Warning: tricontourf failed for t={t_val}: {e}")
                cont = None

            # always create an invisible scatter on top for hover
            scatter = ax.scatter(x_t, y_t, c='none', s=100, picker=True)

          
            crs = mplcursors.cursor(scatter, hover=True)
            @crs.connect("add")
            def on_add(sel):
                i = sel.index
                sel.annotation.set_text(
                    f"x = {x_t[i]:.3f}\n"
                    f"y = {y_t[i]:.3f}\n"
                    f"True T = {temp_t[i]:.2f} °C\n"
                    f"Pred T = {pred_t[i]:.2f} °C\n"
                    f"Δ = {diff_t[i]:.2f}"
                )

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f't={t_val:.3f}')
        
        # Hide unused subplots
        for i in range(len(selected_times), len(axes)):
            axes[i].axis('off')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        # Make sure cont is defined even if there was a runtime error
        if cont is None and i > 0:
            for j in range(i):
                if hasattr(axes[j], 'collections') and axes[j].collections:
                    cont = axes[j].collections[0]
                    break
        
        if cont is not None:
            cbar = fig.colorbar(cont, cax=cbar_ax)
            cbar.set_label('Difference (Prediction - Ground Truth)')
        
        plt.tight_layout(rect=[0, 0, 0.9, 1]) 
        plt.suptitle('Difference between Prediction and Ground Truth over Time')
        
        # Enable interactive mode for better hover behavior
        plt.ion()
        plt.show()
        
        # plt.savefig('time_difference_grid.png', dpi=300, bbox_inches='tight')