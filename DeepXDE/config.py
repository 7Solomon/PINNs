from dataclasses import asdict, dataclass, field, fields
import json
import math
from typing import Optional
from pydantic import BaseModel
from typing import Any
from utils.metadata import BConfig
from utils.dynamic_loss import DynamicLossWeightCallback
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



####
##  Moisture
###
class Richards1DConfig(BaseModel, BConfig):
    input_dim: int = 2
    output_dim: int = 1

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    #decay: list[str,int,float] = ['step', 2000, 0.5]
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE', 'Initial', 'Left', 'Right']
    callbacks: list[str] = ['resample']
    
    #fourier_transform_features: int = 20
    #callbacks: list[str] = ['dynamicLossWeight']

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

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    decay: list[str,int,float] = None
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE_X', 'PDE_Y', 'Left', 'Right']

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

steadyHeatConfig = SteadyHeatConfig()
bernoulliBalkenConfig = BernoulliBalkenConfig()
bernoulliBalken2DConfig = BernoulliBalken2DConfig()
bernoulliBalkenTConfig = BernoulliBalkenTconfig()
cooksMembranConfig = CooksMembranConfig()
transientHeatConfig = TransientHeatConfig()
richards1DConfig = Richards1DConfig()
richardsMixed1DConfig = RichardsMixed1DConfig()
darcy2DConfig = Darcy2DConfig()
thermalMechanical2DConfig = ThermalMechanical2DConfig()
thermalMoisture2DConfig = ThermalMoisture2DConfig()


#concreteData = ConcreteData()