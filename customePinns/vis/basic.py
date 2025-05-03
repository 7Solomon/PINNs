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
def get_time_matrix(device, t, n_grid_points):
  T = np.linspace(t[0],t[1],n_grid_points)
  T_tensor = torch.tensor(T, dtype=torch.float32).to(device)
  return T, T_tensor

def get_pred_grid(model, device,x,y,t,rescale_function, n_grid_points=100):
  X,Y,XY_tensor = get_grid_matrix(device, x, y, n_grid_points)
  T,T_tensor = get_time_matrix(device, t, n_grid_points)
  
  num_xy = XY_tensor.shape[0] # N*N
  num_t = T_tensor.shape[0] # N
  XY_repeated = XY_tensor.repeat_interleave(num_t, dim=0) # Shape [N*N*N, 2]
  T_repeated = T_tensor.unsqueeze(1).repeat(num_xy, 1) # [N*N*N, 1]
  XYt_tensor = torch.cat((XY_repeated, T_repeated), dim=1)
  
  with torch.no_grad():
    pred = model(XYt_tensor).cpu()
    scaled = rescale_function(pred)
    print('scaled', scaled.shape)

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
