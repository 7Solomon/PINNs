import deepxde as dde
import numpy as np

class SlowPdeLossWeightCallback(dde.callbacks.Callback):
    def __init__(self, pde_indices, max_epoch=50000, final_weight=1.0):
        """
        Initializes a callback to slowly increase specific loss weights over time.

        Args:
            pde_indices (int or list[int]): The index or list of indices for the PDE losses.
            max_epoch (int): The number of epochs over which to anneal the weights.
            final_weight (float): The final weight value for the PDE losses.
        """
        super().__init__()
        if not isinstance(pde_indices, list):
            pde_indices = [pde_indices]
        self.pde_indices = pde_indices
        self.max_epoch = max_epoch
        self.final_weight = final_weight
        self.initial_weights = None
    
    def on_train_begin(self):
        """Called at the beginning of training to store initial weights."""
        self.initial_weights = list(self.model.loss_weights)


    def on_epoch_end(self):
        epoch = self.model.train_state.epoch
        
        # Create a mutable copy to update
        current_weights = list(self.model.loss_weights)
        new_weight_value = None

        for index in self.pde_indices:            
            initial_weight = self.initial_weights[index]
            
            if epoch < self.max_epoch:
                progress = epoch / self.max_epoch
                new_weight_value = initial_weight + (self.final_weight - initial_weight) * progress
            else:
                new_weight_value = self.final_weight

            current_weights[index] = new_weight_value
        
        self.model.loss_weights = current_weights
        
        if epoch > 0 and epoch % 1000 == 0 and new_weight_value is not None:
            print(f"\nEpoch {epoch}: Annealing loss weights {self.pde_indices} to {new_weight_value:.4f}")


class DynamicLossWeightCallback(dde.callbacks.Callback):
    def __init__(self, freq=1000, alpha=0.9):
        """
        Initializes the dynamic loss weighting callback. This callback adjusts the loss
        weights during training to ensure that all loss components have a similar magnitude,
        preventing any single loss from dominating the training process.

        Args:
            freq (int): The frequency (in epochs) at which to update the weights.
            alpha (float): The momentum parameter for the exponential moving average of the losses.
                           A higher value gives more weight to past loss values, making the
                           updates smoother but less responsive.
        """
        super().__init__()
        self.freq = freq
        self.alpha = alpha
        self.moving_avg_losses = None

    def on_train_begin(self):
        """Called at the beginning of training."""
        self.num_losses = len(self.model.loss_weights)
        self.moving_avg_losses = None
        print(f"DynamicLossWeightCallback: Initialized for {self.num_losses} loss terms. Updating every {self.freq} epochs.")

    def on_epoch_end(self):
        """Called at the end of each epoch."""
        epoch = self.model.train_state.epoch
        if epoch > 0 and epoch % self.freq == 0:
            self._update_weights()

    def _update_weights(self):
        """
        Updates the loss weights based on the moving average of the unweighted loss values.
        """
        # Get the latest WEIGHTED loss values from the training state
        latest_weighted_losses = np.array(self.model.train_state.loss_train)
        current_weights = np.array(self.model.loss_weights)
        
        if len(latest_weighted_losses) != self.num_losses:
            print("DynamicLossWeightCallback: Warning! Mismatch in number of losses.")
            return

        unweighted_losses = latest_weighted_losses / (current_weights + 1e-8)

        if self.moving_avg_losses is None:
            self.moving_avg_losses = unweighted_losses
        else:
            self.moving_avg_losses = self.alpha * self.moving_avg_losses + (1 - self.alpha) * unweighted_losses

        mean_loss = np.mean(self.moving_avg_losses)
        new_weights = mean_loss / (self.moving_avg_losses + 1e-8)

        self.model.loss_weights = new_weights
        
        if self.model.train_state.epoch % self.freq == 0:
            print(f"\nEpoch {self.model.train_state.epoch}: Updated loss weights to:")
            print([f"{w:.4f}" for w in new_weights])
            print(f"Current unweighted losses:")
            print([f"{l:.4e}" for l in unweighted_losses])