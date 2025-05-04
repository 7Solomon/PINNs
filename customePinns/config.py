from dataclasses import dataclass, field
import torch
import torch.nn as nn
from vars import device 

@dataclass
class HeatConfig:
    alpha: float = 7.5e-7  # Thermal diffusivity
    T_max: float = 100.0
    T_min: float = 0.0
    layers: list[int] = field(default_factory=lambda: [2, 64, 64, 64, 1])
    lambda_ic: float = 1.0
    lambda_bc: float = 1.0
    lambda_pde: float = 1.0
    mse_loss: nn.Module = field(default_factory=nn.MSELoss)
    lr: float = 1e-3
    epochs: int = 300
    model_path: str = 'models/heat'
    device: torch.device = device

@dataclass
class MoistureConfig:
    # Van Genuchten params
    theta_r: float = 0.0
    theta_s: float = 0.1
    alpha_vg: float = 0.4 # Renamed to avoid clash with heat alpha
    n_vg: float = 1.2
    m_vg: float = field(init=False) # Calculated field
    K_s: float = 10e-6
    # PINN params
    layers: list[int] = field(default_factory=lambda: [3, 64, 64, 64, 1])
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
    device: torch.device = device

    def __post_init__(self):
        # Calculate m after initialization
        self.m_vg = 1.0 - (1.0 / self.n_vg)

# You could add more configurations here (e.g., TransientHeatConfig)