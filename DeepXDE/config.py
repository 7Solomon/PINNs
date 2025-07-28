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
    
    #compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    compile_args: dict = {'optimizer': 'L-BFGS', 'lr': 1e-3}
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE','Initial', 'Left', 'Right', 'Top', 'Bottom']

    #decay: list[str,int,float] = ['step', 1000, 0.9]

    callbacks: list[str] = ['dataCollector'] #, 'resample'

    #fourier_transform_features: int = 32



####
##  Moisture
###
class Richards1DConfig(BaseModel, BConfig):
    input_dim: int = 2
    output_dim: int = 1
    activation: str = 'tanh'
    initializer: str = 'Glorot uniform'

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    #compile_args: dict = {'optimizer': 'L-BFGS', 'lr': 1e-3}
    decay: list[str,int,float] = ['step', 1000, 0.9]
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE', 'Initial', 'Left', 'Right']
    callbacks: list[str] = ['resample', 'dataCollector']

    fourier_transform_features: int = 32

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

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    #compile_args: dict = {'optimizer': 'L-BFGS'}
    decay: list[str,int,float] = ['step', 1000, 0.7]
    #loss_weights: list[float] = [1.0, 1.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    loss_weights: list[float] = [1.0, 1.0,  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    #loss_weights: list[float] = [1.0, 1.0,  10.0, 10.0, 1.0]
    #loss_labels: list[str] = ['PDE_X', 'PDE_Y', 'Left_x', 'Left_y', 'Right_no_traction_x', 'Right_no_traction_y', 'Right_xy', 'Top_no_traction_x', 'Top_no_traction_y', 'Bottom_no_traction_x', 'Bottom_no_traction_y']
    loss_labels: list[str] = ['PDE_X', 'PDE_Y', 'Left_x', 'Left_y', 'Right_y', 'Top_no_traction_x', 'Top_no_traction_y', 'Bottom_no_traction_x', 'Bottom_no_traction_y']
    #loss_labels: list[str] = ['PDE_X', 'PDE_Y', 'Left_x', 'Left_y', 'Right_y']
    
    callbacks: list[str] = ['dataCollector']#  , 'slowPdeAnnealing' 

    #fourier_transform_features: int = 64

    #pde_indices: list[int] = [0, 1]
    #annealing_value: float = 1.0
class BernoulliBalken2DEnsembleConfig(BaseModel, BConfig):
    input_dim: int = 2
    output_dim: int = 8
    activation: str = 'tanh'
    initializer: str = 'Glorot uniform'
    layers: list[int] = [100]*6

    #compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    #decay: list[str,int,float] = ['step', 1000, 0.7]

    compile_args: dict = {'optimizer': 'L-BFGS'}
    #loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    #loss_weights: list[float] = [
    #1e-4, 1e-4,          # Equilibrium equations
    #1e-4, 1e-4, 1e-4,    # Consistency equations
    #100.0, 100.0,        # Left displacement BCs
    #1.0, 1.0, 1.0,       # Right stress BCs
    #1.0, 1.0,            # Top stress BCs
    #1.0, 1.0             # Bottom stress BCs
    #]
    loss_weights: list[float] = [
        1.0, 1.0,  # E1, E2 - Equilibrium equations
        1, 1, 1,  # E3, E4, E5 - Hooke’s law residuals for sxx, syy, sxy
        1.0, 1.0, 1.0,  # E6, E7, E8 - Strain-displacement consistency (exx, eyy, exy)

        1.0, 1.0,
        1.0,
        1.0, 1.0,
        1.0, 1.0,
    ]
    loss_labels: list[str] = [
        'PDE_equilibrium_x', 'PDE_equilibrium_y', 
        'PDE_Constitutive_xx', 'PDE_Constitutive_yy', 'PDE_Constitutive_xy',
        'PDE_Strain-Displacement_exx', 'PDE_Strain-Displacement_eyy', 'PDE_Strain-Displacement_exy',
        'bc_left_u_x', 'bc_left_u_y', 
        'bc_right_u_y', 
        'bc_top_sigma_yy', 'bc_top_tau_xy', 
        'bc_bottom_sigma_yy', 'bc_bottom_tau_xy'
    ]

    callbacks: list[str] = ['dataCollector', 'resample'] #,
    
    
    fourier_transform_features: int = 128
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
    activation: str = 'Tanh'
    initializer: str = 'Glorot uniform'
    callbacks: list[str] = ['dataCollector', 'multiHeadScheduler'] #'resample' , 'slowPdeAnnealing',

    model_type: str = 'MultiBranch'
    head_definitions: list = [
                [64, 64, 64, 2],  # Head for u, v
                [64, 64, 64, 1]   # Head for T
            ]

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    decay: list[str,int,float] = ['step', 2500, 0.7]

    #compile_args: dict = {'optimizer': 'L-BFGS'}
    loss_weights: list[float] = [
        1,    # PDE_X
        1,    # PDE_Y
        1,    # PDE_T
        1.0,  # Left_T
        1.0,  # Right_T
        10.0,  # Bottom_U
        10.0,  # Bottom_V
        1.0,  # Initial_T
        #1.0,  # initial_U
        #1.0   # initial_V
    ]
    #loss_labels: list[str] = ['PDE_X', 'PDE_Y', 'PDE_T', 'Left_T', 'Right_T', 'Bottom_U', 'Bottom_V', 'Initial_T', 'initial_U', 'initial_V']
    loss_labels: list[str] = ['PDE_X', 'PDE_Y', 'PDE_T', 'Left_T', 'Right_T', 'Bottom_U', 'Bottom_V']
    head_to_pde_map: list[list] = [[0,1,5,6], [2,3,4]]  
    multi_head_schedule: list[dict] = [
        {'epoch': 0,    'active_heads': [1]},
        {'epoch': 1000, 'active_heads': [0]},
        {'epoch': 2000, 'active_heads': [0]},
        {'epoch': 3000, 'active_heads': [0, 1]},
    ]
    #epoch_etappen: list[int] = [0, 2000, 2000]

    #fourier_transform_features: int = 64
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