from matplotlib import pyplot as plt
from nn_stuff.temp_pinn import PINN
from utils import re_scale_temp
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


def get_T_pred_grid(model: PINN, device,x,y,t, n_grid_points=100):
  X,Y,XY_tensor = get_grid_matrix(device, x, y, n_grid_points)
  T = torch.full((XY_tensor.shape[0], 1), t, dtype=torch.float32).to(device)
  XYt_tensor = torch.cat((XY_tensor, T), dim=1)
 
  with torch.no_grad():
    T_pred = model(XYt_tensor).cpu().numpy()
    T_test = re_scale_temp(T_pred)

  T_grid = T_test.reshape(X.shape)
  return{
      'X':X,
      'Y':Y,
      'T': T, 
      'T_grid':T_grid,
  }

def get_steady_T_pred_grid(model, device, x,y, n_grid_points=100):
  X,Y,XY_tensor = get_grid_matrix(device, x, y, n_grid_points)
  with torch.no_grad():
    T_pred = model(XY_tensor).cpu().numpy()
    T_test = re_scale_temp(T_pred)

  T_grid = T_test.reshape(X.shape)
  return{
      'X':X,
      'Y':Y,
      'T_grid':T_grid,
  }

def vis_plate(data):
  plt.figure(figsize=(10,10))
  plt.contourf(data['X'],data['Y'],data['T_grid'], levels=100,cmap='jet')
  plt.colorbar()
  plt.show()

def vis_csv():
  pass

