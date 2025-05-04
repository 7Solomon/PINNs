from matplotlib import pyplot as plt
import numpy as np
import torch

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
