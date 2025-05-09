from matplotlib import pyplot as plt
from mechanic.domain import scale_tensor
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
  else:
    plt.plot(Loss)
  
  plt.grid(True, which="both", ls="--")
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Loss (log)')
  plt.yscale('log')
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
    pred = model(ZT_tensor)
    for i, _ in enumerate(pred):
      pred[i] = rescale_function(_)
      pred[i].numpy().reshape(Z.shape)

  return {
      'Z': Z,
      'T': T,
      'output': pred,
  }


def vis_x_q(model, x):
  with torch.no_grad():
    pred = model(x).cpu()
    pred = pred.numpy()
  plt.plot(x[:,0].cpu().detach().numpy(), pred, label='Predicted')
  plt.xlabel('X')
  plt.ylabel('Predicted Value')
  plt.title('Model Prediction vs X')
  plt.legend()
  plt.grid(True)
  plt.show()

def visualize_beam_deflection(model, domain, num_points=100,analytical_func=None):
    # Create evenly spaced points for smooth visualization
    x_min, x_max = domain.dimension['x']
    x_test = torch.linspace(x_min, x_max, num_points).reshape(-1, 1)
    q_val = torch.ones_like(x_test)* 10.0 #domain.collocation[0, -1].item()  # Extract the q value from collocation points

    
    #get scaling
    scaled_x_test = x_test * domain.scale['l']
    scaled_q_val = (q_val *  (domain.scale['E'] * domain.scale['I'])) / (domain.scale['l']**3)
    #scaled_x_test = scale_tensor(x_test, domain.scale_domains['collocation']['points'])
    #scaled_q_val = scale_tensor(q_val, domain.scale_domains['collocation']['inputs'])
    
    
    # Create input tensor with correct format
    x_input = torch.zeros((num_points, 2))
    x_input[:, 0] = scaled_x_test.squeeze()
    x_input[:, 1] = scaled_q_val.squeeze()
    print(f'x_input: {x_input[:5]}')
    print(f'q_val: {q_val[:2]}')

    # Make prediction
    with torch.no_grad():
        pred = model(x_input).cpu().numpy()
    
    print(f'{pred[:5], pred[-5:]}')
    #pred = pred * domain.scale['l']
    #print(pred[:2])
    #
    ## Setup figure with improved styling
    #plt.figure(figsize=(10, 6))
    #x_np = x_test.cpu().numpy()
    #plt.plot(x_np, pred, 'b-', linewidth=2, label='pred')
    #
    ## Ground
    #if analytical_func is not None:
    #    analytical = analytical_func(x_np, q_val, (x_max-x_min), 1)
    #    plt.plot(x_np, analytical, 'r--', linewidth=2, label='Analytical Solution')
    #    
    #    # Calculate and display error metrics
    #    #mse = np.mean((pred - analytical)**2)
    #    #plt.text(0.05, 0.95, f'MSE: {mse:.2e}', transform=plt.gca().transAxes, 
    #    #         bbox=dict(facecolor='white', alpha=0.8))
    #
    ## Plot boundary conditions as points
    #if hasattr(domain, 'boundarys') and domain.boundarys:
    #    for key, condition in domain.boundarys.items():
    #        if 'x_min' in key:
    #            plt.scatter(condition.points[:, 0].detach().cpu(), condition.values[:,0].detach().cpu(), 
    #                       color='green', s=50, label='Boundary Condition')
    #            break
    #
    #plt.xlabel('X', fontsize=12)
    #plt.ylabel('Verschiebung (w)', fontsize=12)
    #plt.title(f'Verschiebung unter Constant Load q={q_val}', fontsize=14)
    #plt.grid(True, linestyle='--', alpha=0.7)
    #plt.legend(fontsize=10)
    #
    #plt.plot([x_min, x_max], [0, 0], 'k-', linewidth=3, label='Beam Unverformt')
    #
    #arrow_count = 10
    #arrow_x = np.linspace(x_min, x_max, arrow_count)
    #arrow_start_y = 0.05 * (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0])
    #arrow_len = 0.04 * (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0])
    #for x in arrow_x:
    #    plt.arrow(x, arrow_start_y, 0, -arrow_len, 
    #             head_width=0.02, head_length=0.3*arrow_len, fc='r', ec='r')
    #y_min = min(np.min(pred), 0) * 1.1  # 
    #y_max = max(arrow_start_y + 0.02, np.max(pred) * 1.1)
    #plt.ylim(y_min, y_max)
    #
    #plt.tight_layout()
    #plt.show()