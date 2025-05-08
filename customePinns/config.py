from dataclasses import dataclass, field
import torch
import torch.nn as nn

from heat.nn_stuff.train import train_steady_loop, train_transient_loop
from moisture.normal_pinn_2d.nn_stuff.train import train_moisture_loop
from moisture.head_body_pinn_1d.nn_stuff.train import train_body_head_loop
from mechanic.nn_stuff.train import train_bernouli_balken_loop

from nn_stuff.pinn import PINN, BodyHeadPINN

@dataclass
class BConfig:
    device: torch.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def type(self) -> str:
        return self.__class__.__name__.lower()


@dataclass
class SteadyHeatConfig(BConfig):
    PINN_type: PINN = PINN
    pinn_creation_vars: list[list[int]] = field(default_factory=lambda: [[2, 64, 64, 64, 1]]) #  layers
    alpha: float = 7.5e-7  # Thermal diffusivity
    
    T_max: float = 100.0
    T_min: float = 0.0
    lambda_ic: float = 1.0
    lambda_bc: float = 1.0
    lambda_pde: float = 1.0
    mse_loss: nn.Module = field(default_factory=nn.MSELoss)
    lr: float = 1e-4
    epochs: int = 5000
    model_path: str = 'models/heat'
    train_loop: callable = train_steady_loop

@dataclass
class TransientHeatConfig(BConfig):
    PINN_type: PINN = PINN
    pinn_creation_vars: list[list[int]]= field(default_factory=lambda: [[3, 64, 64, 64, 1]]) #  layers

    alpha: float = 7.5e-7  
    T_max: float = 100.0
    T_min: float = 0.0
    lambda_ic: float = 1.0
    lambda_bc: float = 1.0
    lambda_pde: float = 1.0
    mse_loss: nn.Module = field(default_factory=nn.MSELoss)
    lr: float = 1e-3
    epochs: int = 300
    model_path: str = 'models/heat'
    train_loop: callable = train_transient_loop


@dataclass
class MoistureConfig(BConfig):
    PINN_type: PINN = PINN
    pinn_creation_vars: list[list[int]] = field(default_factory=lambda: [[3, 64, 64, 64, 1],]) #  layers
  

    # Van Genuchten params
    theta_r: float = 0.0
    theta_s: float = 0.1
    alpha_vg: float = 0.4
    n_vg: float = 1.2
    m_vg: float = field(init=False) # Calculated field
    K_s: float = 10e-6
    # PINN params
    lambda_ic: float = 1.0
    lambda_bc: float = 1.0
    lambda_pde: float = 1e9
    mse_loss: nn.Module = field(default_factory=nn.MSELoss)
    lr: float = 5e-6
    epochs: int = 1000
    max_norm: float = 1.0
    model_path: str = 'models/moisture' 
    h_max: float = 60.0
    h_min: float = 0.0
    train_loop: callable = train_moisture_loop

    def __post_init__(self):
        self.m_vg = 1.0 - (1.0 / self.n_vg)

@dataclass
class MoistureHeadBodyConfig(BConfig):
    PINN_type: BodyHeadPINN = BodyHeadPINN
    pinn_creation_vars: list[list[int]] = field(default_factory=lambda: [[2, 64, 64, 64, 1], 
                                                                         [1, 64, 64, 1]]) #  layers, body head
    z_range: tuple[float, float] = (0, 2)
    t_range: tuple[float, float] = (0, 10)
    lambda_ic: float = 1.0
    lambda_bc: float = 1.0
    lambda_pde: float = 1e9
    mse_loss: nn.Module = field(default_factory=nn.MSELoss)
    lr: float = 5e-7
    epochs: int = 1000
    max_norm: float = 1.0
    model_path: str = 'models/moisture_1d' 
    h_max: float = -60.0
    h_min: float = 0.0
    train_loop: callable = train_body_head_loop

@dataclass
class BernoulliBalkenConfig(BConfig):
    PINN_type: PINN = PINN
    pinn_creation_vars: list[list[int]] = field(default_factory=lambda: [[2, 64, 64, 64, 1]])

    EI: float = 1 # Biegesteifigkeit

    lambda_bc: float = 1.0
    lambda_pde: float = 1.0

    lr: float = 1e-4
    epochs: int = 6000

    train_loop: callable = train_bernouli_balken_loop
    model_path: str = 'models/bernouli_balken'

