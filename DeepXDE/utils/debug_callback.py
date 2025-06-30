import deepxde as dde
  #print('-----')
  #print('u', y[:,0].min().item(), y[:,0].max().item())
  #print('v', y[:,1].min().item(), y[:,1].max().item())
  #print('sigmax_x:', sigmax_x_nd.min().item(), sigmax_x_nd.max().item())
  #print('sigmay_y:', sigmay_y_nd.min().item(), sigmay_y_nd.max().item())
  #print('tauxy_y:', tauxy_y_nd.min().item(), tauxy_y_nd.max().item())
  #print('tauxy_x:', tauxy_x_nd.min().item(), tauxy_x_nd.max().item())
  #print('b_force:', b_force)
  #print('scale.sigma:', scale.sigma)
  #print('scale.L:', scale.L)
  #print('scale.U:', scale.U)
  #print()


class DebugCallback(dde.callbacks.Callback):
    def __init__(self):

        super().__init__()
  
    def on_train_begin(self):
        pass
    def on_epoch_end(self):
        epoch = self.model.train_state.epoch

        if epoch > 0 and epoch % 1000 == 0:
            print(f"\nEpoch {epoch}: Annealing loss weights {self.pde_indices} to {new_weight_value:.4f}")
        
