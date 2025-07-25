from dataclasses import asdict, dataclass, field, fields
import json
import math
from typing import Optional
from pydantic import BaseModel
from typing import Any
from utils.metadata import BConfig
import torch

###
#  Heat
###
class SteadyHeatConfig(BaseModel, BConfig):
    input_dim: int = 2
    output_dim: int = 1
    activation: str = 'tanh'
    initializer: str = 'Glorot uniform'

    loss_weights: list[float] = [1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE', 'Left', 'Right']
    decay: Optional[Any] = None

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}

class TransientHeatConfig(BaseModel, BConfig):
    input_dim: int = 3
    output_dim: int = 1
    activation: str = 'tanh'
    initializer: str = 'Glorot uniform'
    
    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE','Initial', 'Left', 'Right', 'Top', 'Bottom']

    #decay: list[str,int,float] = ['step', 1000, 0.9]

    callbacks: list[str] = ['dataCollector']

    #fourier_transform_features: int = 32



####
##  Moisture
###
class Richards1DConfig(BaseModel, BConfig):
    input_dim: int = 2
    output_dim: int = 1
    activation: str = 'tanh'
    initializer: str = 'Glorot uniform'

    compile_args: dict = {'optimizer': 'L-BFGS', 'lr': 1e-3}
    decay: list[str,int,float] = ['step', 1000, 0.9]
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE', 'Initial', 'Left', 'Right']
    callbacks: list[str] = ['resample', 'dataCollector']
    
    #fourier_transform_features: int = 32

class RichardsMixed1DConfig(BaseModel, BConfig):
    input_dim: int = 2
    output_dim: int = 2
    activation: str = 'tanh'
    initializer: str = 'Glorot uniform'


    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE_Head', 'PDE_Saturation', 'Initial', 'Left', 'Right']

class Darcy2DConfig(BaseModel, BConfig):
    input_dim: int = 2
    output_dim: int = 1
    activation: str = 'tanh'
    initializer: str = 'Glorot uniform'

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    loss_weights: list[float] = [1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE', 'Left', 'Right']


###
#  Mechanical
##

class BernoulliBalkenConfig(BaseModel, BConfig):
    input_dim: int = 1
    output_dim: int = 1
    activation: str = 'tanh'
    initializer: str = 'Glorot uniform'

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0]
    #loss_labels: list[str] = ['PDE_X', 'Left', 'Right']

class BernoulliBalken2DConfig(BaseModel, BConfig):
    input_dim: int = 2
    output_dim: int = 2
    activation: str = 'tanh'
    initializer: str = 'Glorot uniform'

    #compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    compile_args: dict = {'optimizer': 'L-BFGS'}
    decay: list[str,int,float] = ['step', 5000, 0.9]
    loss_weights: list[float] = [1.0, 1.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE_X', 'PDE_Y', 'Left_x', 'Left_y', 'Right_no_traction_x', 'Right_no_traction_y', 'Top_no_traction_x', 'Top_no_traction_y', 'Bottom_no_traction_x', 'Bottom_no_traction_y']
    callbacks: list[str] = ['slowPdeAnnealing', 'dataCollector', 'resample']

    fourier_transform_features: int = 64

    pde_indices: list[int] = [0, 1] 
    annealing_value: float = 10.0 
class BernoulliBalken2DEnsembleConfig(BaseModel, BConfig):
    input_dim: int = 2
    output_dim: int = 5
    activation: str = 'tanh'
    initializer: str = 'Glorot uniform'

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    loss_weights: list[float] = [100.0, 100.0, 100.0, 100.0, 10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE_X', 'PDE_Y', 'PDE_sigma_pred_x', 'PDE_sigma_pred_y', 'sigma_x_x_div', 'sigma_y_y_div', 'sigma_xy_y_div', 'sigma_xy_x_div', 'Left_x', 'Left_y', 'Right_traction_x', 'Right_traction_y', 'right_traction_xy', 'Top_no_traction_x', 'Top_no_traction_y', 'Bottom_no_traction_x', 'Bottom_no_traction_y']
    decay: list[str,int,float] = ['step', 1000, 0.9]
    callbacks: list[str] = ['resample', 'dataCollector']
    
    
    fourier_transform_features: int = 64
    #variable_lr_config: dict = {
    #    'mode': 'loss_based', 
    #    'patience': 300,
    #    'factor': 0.7,
    #    'min_lr': 1e-8
    #}

class BernoulliBalkenTconfig(BaseModel, BConfig):
    input_dim: int = 2
    output_dim: int = 1
    activation: str = 'tanh'
    initializer: str = 'Glorot uniform'

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


class CooksMembranConfig(BaseModel, BConfig):
    input_dim: int = 2
    output_dim: int = 2
    activation: str = 'tanh'
    initializer: str = 'Glorot uniform'

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    #loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0]


class ThermalMechanical2DConfig(BaseModel, BConfig):
    input_dim: int = 3
    output_dim: int = 3
    activation: str = 'tanh'
    initializer: str = 'Glorot uniform'
    callbacks: list[str] = ['dataCollector']

    model_type: str = 'MultiBranch'
    branches: dict = {
        'uv_branch': {
            'input_indices': [0, 1],
            'layer_dims': [2, 20, 20, 2]
        },
        't_branch': {
            'input_indices': [2],
            'layer_dims': [1, 20, 20, 1]
        }
    }

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-4}
    #decay: list[str,int,float] = ['exponential', 5000, 0.7]
    loss_weights: list[float] = [
        1.0,    # PDE_X
        1.0,    # PDE_Y
        1.0,    # PDE_T
        100.0,  # Left_T
        100.0,  # Right_T
        100.0,  # Bottom_U
        100.0,  # Bottom_V
        100.0,  # Initial_T
        100.0,  # initial_U
        100.0   # initial_V
    ]
    loss_labels: list[str] = ['PDE_X', 'PDE_Y', 'PDE_T', 'Left_T', 'Right_T', 'Bottom_U', 'Bottom_V', 'Initial_T', 'initial_U', 'initial_V']

class ThermalMoisture2DConfig(BaseModel, BConfig):
    input_dim: int = 3
    output_dim: int = 2
    activation: str = 'tanh'
    initializer: str = 'Glorot uniform'

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-4}
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE', 'temp_init', 'moisture_init', 'temp_left', 'moisture_left']

class MechanicalMoisture2DConfig(BaseModel, BConfig):
    input_dim: int = 3
    output_dim: int = 3
    activation: str = 'tanh'
    initializer: str = 'Glorot uniform'

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-4}
    #decay: list[str,int,float] = ['step', 10000, 0.5]
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE_U', 'PDE_V', 'PDE_Theta', 'inital_theta', 'Left_U', 'Left_V', 'Top_Theta']

steadyHeatConfig = SteadyHeatConfig()
bernoulliBalkenConfig = BernoulliBalkenConfig()
bernoulliBalken2DConfig = BernoulliBalken2DConfig()
bernoulliBalken2DEnsembleConfig = BernoulliBalken2DEnsembleConfig()
bernoulliBalkenTConfig = BernoulliBalkenTconfig()
cooksMembranConfig = CooksMembranConfig()
transientHeatConfig = TransientHeatConfig()
richards1DConfig = Richards1DConfig()
richardsMixed1DConfig = RichardsMixed1DConfig()
darcy2DConfig = Darcy2DConfig()
thermalMechanical2DConfig = ThermalMechanical2DConfig()
thermalMoisture2DConfig = ThermalMoisture2DConfig()
mechanicalMoisture2DConfig = MechanicalMoisture2DConfig()


#concreteData = ConcreteData()