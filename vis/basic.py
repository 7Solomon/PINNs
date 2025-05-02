from matplotlib import pyplot as plt
import numpy as np
import torch

def plot_loss(Loss):
  plt.plot(Loss)
  plt.show()

def get_grid_matrix(device, x, y, n_grid_points):
  x_matrix = np.linspace(x[0],x[1],n_grid_points)
  y_matrix = np.linspace(y[0],y[1],n_grid_points)
  X, Y = np.meshgrid(x_matrix,y_matrix)
  XY = np.vstack([X.ravel(),Y.ravel()]).T
  XY_tensor = torch.tensor(XY, dtype=torch.float32).to(device)
  return X,Y,XY_tensor


def get_pred_grid(model, device,x,y,t,rescale_function, n_grid_points=100):
  X,Y,XY_tensor = get_grid_matrix(device, x, y, n_grid_points)
  T = torch.full((XY_tensor.shape[0], 1), t, dtype=torch.float32).to(device)
  XYt_tensor = torch.cat((XY_tensor, T), dim=1)
 
  with torch.no_grad():
    pred = model(XYt_tensor).cpu().numpy()
    scaled = rescale_function(pred)

  output = scaled.reshape(X.shape)
  return{
      'X':X,
      'Y':Y,
      'T': T, 
      'output':output,
  }

def get_steady_pred_grid(model, device, x,y,rescale_function, n_grid_points=100):
  X,Y,XY_tensor = get_grid_matrix(device, x, y, n_grid_points)
  with torch.no_grad():
    pred = model(XY_tensor).cpu().numpy()
    scaled = rescale_function(pred)

  output = scaled.reshape(X.shape)
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

