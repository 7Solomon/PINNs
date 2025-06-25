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
    loss_weights: list[float] = [1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE', 'Left', 'Right']
    decay: Optional[Any] = None

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}

class TransientHeatConfig(BaseModel, BConfig):
    input_dim: int = 3
    output_dim: int = 1
    
    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE', 'Left', 'Right', 'Initial']

    decay: list[str,int,float] = ['step', 1000, 0.9]



####
##  Moisture
###
class Richards1DConfig(BaseModel, BConfig):
    input_dim: int = 2
    output_dim: int = 1

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    decay: list[str,int,float] = ['step', 1000, 0.9]
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE', 'Initial', 'Left', 'Right']
    callbacks: list[str] = ['resample']
    
    #fourier_transform_features: int = 20

class RichardsMixed1DConfig(BaseModel, BConfig):
    input_dim: int = 2
    output_dim: int = 2

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    loss_weights: list[float] = [1.0, 1.0, 10.0, 10.0, 10.0]
    loss_labels: list[str] = ['PDE_Head', 'PDE_Saturation', 'Initial', 'Left', 'Right']

class Darcy2DConfig(BaseModel, BConfig):
    input_dim: int = 2
    output_dim: int = 1

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    loss_weights: list[float] = [1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE', 'Left', 'Right']


###
#  Mechanical
##

class BernoulliBalkenConfig(BaseModel, BConfig):
    input_dim: int = 1
    output_dim: int = 1

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0]
    #loss_labels: list[str] = ['PDE_X', 'Left', 'Right']

class BernoulliBalken2DConfig(BaseModel, BConfig):
    input_dim: int = 2
    output_dim: int = 2

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-4}
    #decay: list[str,int,float] = ['step', 1000, 0.9]
    loss_weights: list[float] = [100.0, 10.0, 1.0, 1.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    loss_labels: list[str] = ['PDE_X', 'PDE_Y', 'Left_x', 'Left_y', 'Right_no_traction_x', 'Right_no_traction_y', 'Top_no_traction_x', 'Top_no_traction_y', 'Bottom_no_traction_x', 'Bottom_no_traction_y']
    #callbacks: list[str] = ['slowPdeAnnealing']

    #fourier_transform_features: int = 128

    #pde_indices: list[int] = [0, 1] 
    #annealing_value: float = 10.0 
class BernoulliBalken2DEnsembleConfig(BaseModel, BConfig):
    input_dim: int = 2
    output_dim: int = 5

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-4}
    loss_weights: list[float] = [100.0, 100.0, 100.0, 100.0, 10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE_X', 'PDE_Y', 'PDE_sigma_pred_x', 'PDE_sigma_pred_y', 'sigma_x_x_div', 'sigma_y_y_div', 'sigma_xy_y_div', 'sigma_xy_x_div', 'Left_x', 'Left_y', 'Right_traction_x', 'Right_traction_y', 'right_traction_xy', 'Top_no_traction_x', 'Top_no_traction_y', 'Bottom_no_traction_x', 'Bottom_no_traction_y']
    decay: list[str,int,float] = ['step', 1000, 0.9]
    callbacks: list[str] = ['resample']
    fourier_transform_features: int = 128
class BernoulliBalkenTconfig(BaseModel, BConfig):
    input_dim: int = 2
    output_dim: int = 1

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


class CooksMembranConfig(BaseModel, BConfig):
    input_dim: int = 2
    output_dim: int = 2

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    #loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0]


class ThermalMechanical2DConfig(BaseModel, BConfig):
    input_dim: int = 3
    output_dim: int = 3

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-4}
    decay: list[str,int,float] = ['step', 10000, 0.5]
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE_X', 'PDE_Y', 'PDE_T', 'Left_T', 'Right_T', 'Bottom_U', 'Bottom_V', 'Initial_T']

class ThermalMoisture2DConfig(BaseModel, BConfig):
    input_dim: int = 3
    output_dim: int = 2

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-4}
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE', 'temp_init', 'moisture_init', 'temp_left', 'moisture_left']

class MechanicalMoisture2DConfig(BaseModel, BConfig):
    input_dim: int = 3
    output_dim: int = 3

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