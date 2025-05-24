from process.moisture.scale import rescale_h, scale_t, scale_z
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def vis_1d_head(model, domain, interval=200, title='Richards 1d', xlabel='z', ylabel='u(z,t)'):
    """
    Generates an animation of a 1D model's prediction changing over time.

    Args:
        model: The trained model object with a predict(XT) method.
        spatial_domain (tuple): (x_start, x_end, num_x_points) for the spatial domain.
        time_domain (tuple): (t_start, t_end, num_t_points) for the time domain.
        interval (int, optional): Delay between frames in milliseconds. Defaults to 200.
        title (str, optional): Title of the plot. Defaults to '1D Model Prediction Over Time'.
        xlabel (str, optional): Label for the x-axis. Defaults to 'x'.
        ylabel (str, optional): Label for the y-axis. Defaults to 'u(x,t)'.
    """
    z_start, z_end = domain.spatial['z']
    t_start, t_end = domain.temporal['t']
    num_x_points = 100
    num_t_points = 100

    z_points = np.linspace(z_start, z_end, num_x_points)
    t_points = np.linspace(t_start, t_end, num_t_points)

    Z, T = np.meshgrid(z_points, t_points)
    ZT = np.vstack((Z.ravel(), T.ravel())).T

    Z_scaled = scale_z(Z.copy())
    T_scaled = scale_t(T.copy())
    ZT_scaled = np.vstack((Z_scaled.ravel(), T_scaled.ravel())).T


    # Get predictions from the model
    predictions = model.predict(ZT_scaled)
    predictions = rescale_h(predictions)


    if predictions.ndim > 1 and predictions.shape[1] > 1:
        print(f"Warning: Model output has shape {predictions.shape}. Assuming the first column is the desired output.")

    try:
        data = predictions.reshape(num_t_points, num_x_points).T
    except ValueError as e:
        print(f"Error reshaping predictions: {e}")
        print(f"Prediction shape: {predictions.shape}, Target shape: ({num_t_points}, {num_x_points}) then transposed.")
        print("Ensure the total number of elements in predictions matches num_x_points * num_t_points.")
        return None


    fig, ax = plt.subplots()
    line, = ax.plot(z_points, data[:, 0])

    ax.set_xlim(z_points.min(), z_points.max())
    ax.set_ylim(data.min(), data.max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    def update(frame):
        line.set_ydata(data[:, frame])
        ax.set_title(f'{title} (t={t_points[frame]:.2f})')
        return line,

    ani = animation.FuncAnimation(fig, update, frames=num_t_points,
                                  interval=interval, blit=True, repeat=False)
    
    ani.save('animation.mp4', writer='ffmpeg', fps=1000/interval)
    #plt.show() # geht glaube nocht auf ssh
    return ani
