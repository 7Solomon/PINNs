import deepxde as dde
import numpy as np
import torch

from utils.metadata import BSaver

class DataCollectorCallback(dde.callbacks.Callback):
    def __init__(self, points_data: dict, scale: BSaver, gnd_truth: np.ndarray):
        super().__init__()
        print('DataCollectorCallback initialized')
        #print('points_data', points_data.__contains__('spacetime_points_flat'), points_data.__contains__('spatial_points_flat'))
        self.points_data = points_data
        self.test_points = points_data.get('spacetime_points_flat', points_data.get('spatial_points_flat', None))
        #print('test_points', self.test_points.shape if self.test_points is not None else None)
        self.gnd_truth = gnd_truth / np.array(scale.value_scale_list)
        self.collected_data = {
            'learning_rates': [],
            'loss_weights_history': [],
            'mse_history': [],
        }
    
    def on_epoch_end(self):
        epoch = self.model.train_state.epoch
        
        # Collect learning rate
        if hasattr(self.model.opt, '_learning_rate'):
            self.collected_data['learning_rates'].append(float(self.model.opt._learning_rate))
        
        self.collected_data['loss_weights_history'].append(list(self.model.loss_weights))
        if (epoch > 0 and epoch % 1000 == 0) or epoch == 0:
            predictions = self.model.predict(self.test_points)
            print(predictions.shape, self.gnd_truth.shape)
            predictions = self.points_data['reshape_utils']['pred_to_ij'](predictions)
            print(predictions.shape, self.gnd_truth.shape)


            test = predictions - self.gnd_truth
            print('test', test.min(), test.max(), test.mean(), test.std())

            # Check if the number of points match before calculating MSE
            if self.gnd_truth.shape == predictions.shape:
                mse = dde.metrics.mean_squared_error(self.gnd_truth.ravel(), predictions.ravel())
                print(f"Epoch {epoch}: MSE = {mse:.4f}")
                self.collected_data['mse_history'].append(mse)
            else:
                print(f"Epoch {epoch}: Shape mismatch. GND truth has {self.gnd_truth.shape}, but predictions have {predictions.shape} points. MSE not calculated.")