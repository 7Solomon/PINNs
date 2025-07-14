import deepxde as dde
import numpy as np
import torch

from utils.metadata import BSaver
#from process.heat.vis import vis_transient_field_test
from process.moisture.vis import vis_1d_saturation_test, vis_1d_head_test, vis_1d_time_plot
from process.mechanic.vis import visualize_field_2d_test
from domain_vars import einspannung_2d_domain

class DataCollectorCallback(dde.callbacks.Callback):
    def __init__(self, points_data: dict, scale: BSaver, gnd_truth: np.ndarray):
        super().__init__()
        print('DataCollectorCallback initialized')
        self.points_data = points_data
        self.test_points = points_data.get('spacetime_points_flat', points_data.get('spatial_points_flat', None)) / scale.input_scale_list
        self.scale = scale
        self.gnd_truth = gnd_truth

        self.collected_data = {
            'learning_rates': [],
            'loss_weights_history': [],
            'mse_history': [],
        }
        
        self.callback_count = 0
        self.is_lbfgs = False
    
    def _detect_optimizer(self):
        """Detect if we're using L-BFGS optimizer"""
        if hasattr(self.model, 'opt'):
            opt_name = str(type(self.model.opt).__name__).upper()
            self.is_lbfgs = 'LBFGS' in opt_name or 'L-BFGS' in opt_name
            print(f"Detected optimizer: {opt_name}, L-BFGS mode: {self.is_lbfgs}")
        return self.is_lbfgs
    
    def on_train_begin(self):
        """detect optimizer type"""
        self._detect_optimizer()
    
    def on_epoch_end(self):
        epoch = self.model.train_state.epoch
        self.callback_count += 1
        
        # Collect learning rate
        if hasattr(self.model.opt, '_learning_rate'):
            self.collected_data['learning_rates'].append(float(self.model.opt._learning_rate))
        
        self.collected_data['loss_weights_history'].append(list(self.model.loss_weights))


        # Get MSE frequency
        COLLECT = False
        if self.is_lbfgs:
            COLLECT = True
            print(f"L-BFGS callback #{self.callback_count}, epoch {epoch}")
        else:
            COLLECT = (epoch > 0 and epoch % 50 == 0)


        if COLLECT:
            predictions = self.model.predict(self.test_points)
            predictions = self.points_data['reshape_utils']['pred_to_ij'](predictions)
            predictions = predictions * self.scale.value_scale_list
            
            if self.gnd_truth.shape == predictions.shape:
                mse = dde.metrics.mean_squared_error(self.gnd_truth.flatten(), predictions.flatten())
                self.collected_data['mse_history'].append(mse)
                if self.is_lbfgs:
                    print(f"MSE collected: {mse:.6e}")
            else:
                print(f"Epoch {epoch}: Shape mismatch. GND truth has {self.gnd_truth.shape}, but predictions have {predictions.shape} points. MSE not calculated.")