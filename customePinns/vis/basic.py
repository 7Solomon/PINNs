from matplotlib import pyplot as plt
import numpy as np
import torch

from moisture.normal_pinn_2d.physics import S_e, WRC, HC

def plot_loss(Loss):
  if isinstance(Loss, dict):
    for element in Loss.keys():
      if isinstance(Loss[element], dict):
        for key in Loss[element].keys():
          print(f'{element}[{key}]: {Loss[element][key][:5]} with {len(Loss[element][key])}')
          plt.plot(Loss[element][key], label=f'{element}[{key}]')
      else:
        plt.plot(Loss[element], label=element)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss (log)')
    plt.yscale('log')
    plt.show()
  else:
    plt.plot(Loss)
    plt.show()

def get_grid_matrix(device, x, y, n_grid_points):
  x_matrix = np.linspace(x[0],x[1],n_grid_points)
  y_matrix = np.linspace(y[0],y[1],n_grid_points)
  X, Y = np.meshgrid(x_matrix,y_matrix)
  XY = np.vstack([X.ravel(),Y.ravel()]).T
  XY_tensor = torch.tensor(XY, dtype=torch.float32).to(device)
  return X,Y,XY_tensor
def get_xyt_matrix(device, x, y, t, n_grid_points): 
  x_matrix = np.linspace(x[0],x[1],n_grid_points)
  y_matrix = np.linspace(y[0],y[1],n_grid_points)
  t_matrix = np.linspace(t[0],t[1],n_grid_points)
  X, Y, T = np.meshgrid(x_matrix,y_matrix,t_matrix)
  XYT = np.vstack([X.ravel(),Y.ravel(),T.ravel()]).T
  XYT_tensor = torch.tensor(XYT, dtype=torch.float32).to(device)
  return X,Y,T,XYT_tensor
def get_pred_grid(model, device,x,y,t,rescale_function, n_grid_points=100):
  X,Y,T,XYt_tensor = get_xyt_matrix(device, x, y, t, n_grid_points)
  with torch.no_grad():
    pred = model(XYt_tensor).cpu()
    scaled = rescale_function(pred)
    
  output = scaled.numpy().reshape(X.shape)
  return{
      'X':X,
      'Y':Y,
      'T': T, 
      'output':output,
  }

def get_steady_pred_grid(model, device, x,y,rescale_function, n_grid_points=100):
  X,Y,XY_tensor = get_grid_matrix(device, x, y, n_grid_points)
  with torch.no_grad():
    pred = model(XY_tensor).cpu()
    scaled = rescale_function(pred)

  output = scaled.numpy().reshape(X.shape)
  return{
      'X':X,
      'Y':Y,
      'T_grid':output,
  }

def vis_plate_2d(data):
  plt.figure(figsize=(10,10))
  plt.contourf(data['X'],data['Y'],data['T_grid'], levels=100,cmap='jet')
  plt.colorbar()
  plt.show()


def plot_saturation(conf):
    h = torch.linspace(-2000, 50, steps=500)
    S_e_vals = S_e(h, conf).detach().numpy()
    theta_vals = WRC(h, conf).detach().numpy()
    K_vals = HC(h, conf).detach().numpy()

    plt.figure(figsize=(10, 10))
    
    plt.subplot(1, 3, 1)
    plt.plot(h, S_e_vals, label='Saturation S_e', color='blue')
    plt.xlabel('Pressure head (h) [cm]')
    plt.ylabel('S_e')

    plt.subplot(1, 3, 2)
    plt.plot(h, theta_vals, label='Water retention Curve', color='green')
    plt.xlabel('Pressure head (h) [cm]')
    plt.ylabel('theta')

    plt.subplot(1, 3, 3)
    plt.plot(h, K_vals, label='Hydraulic Conductivity', color='red')
    plt.xlabel('Pressure head (h) [cm]')
    plt.ylabel('K')

    plt.tight_layout()
    plt.show()


## for 1d head body stuff
def get_zt_matrix(device, z, t, n_grid_points):
  """Generates a 2D grid and tensor for Z and T coordinates."""
  z_matrix = np.linspace(z[0], z[1], n_grid_points)
  t_matrix = np.linspace(t[0], t[1], n_grid_points)
  Z, T = np.meshgrid(z_matrix, t_matrix, indexing='ij') # Use 'ij' indexing for consistency if Z is the first dimension
  ZT = np.vstack([Z.ravel(), T.ravel()]).T
  ZT_tensor = torch.tensor(ZT, dtype=torch.float32).to(device)
  return Z, T, ZT_tensor

def get_zt_pred_grid(model, device, z, t, rescale_function, n_grid_points=100):
  """
  Generates predictions on a 2D grid defined by Z and T coordinates.

  Args:
      model: The trained PyTorch model.
      device: The torch device ('cuda' or 'cpu').
      z: A tuple or list defining the range for the Z coordinate (z_min, z_max).
      t: A tuple or list defining the range for the T coordinate (t_min, t_max).
      rescale_function: A function to rescale the model's output.
      n_grid_points: The number of points along each axis of the grid.

  Returns:
      A dictionary containing the Z grid, T grid, and the model's scaled output
      reshaped to the grid dimensions.
  """
  Z, T, ZT_tensor = get_zt_matrix(device, z, t, n_grid_points)
  with torch.no_grad():
    # Assuming the model expects input shape (N, 2) where columns are (z, t)
    pred = model(ZT_tensor)
    scaled = rescale_function(pred[0])

  # Reshape the output to match the grid dimensions (n_grid_points, n_grid_points)
  output = scaled.numpy().reshape(Z.shape)
  return {
      'Z': Z,
      'T': T,
      'output': output,
  }

