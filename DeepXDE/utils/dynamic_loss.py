import deepxde as dde
import numpy as np

class DynamicLossWeightCallback(dde.callbacks.Callback):
    def __init__(self, model, pde_index, max_epoch=10000):
        self.model = model
        self.pde_index = pde_index
        self.max_epoch = max_epoch

    
    def on_epoch_end(self):
        epoch = self.model.train_state.epoch
        new_weight = min(1.0, epoch / self.max_epoch)

        current_weights = list(self.model.loss_weights)
        current_weights[self.pde_index] = new_weight
        self.model.loss_weights = current_weights
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Updated loss weight[{self.pde_index}] to {new_weight:.4f}")