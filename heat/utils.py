from dataclasses import dataclass, field
import enum
import numpy as np
import torch

from heat.vars import *

def norm_temp(T: torch.Tensor) -> torch.Tensor:
  return (T - T_min) / (T_max - T_min)

def re_scale_temp(T: torch.Tensor) -> torch.Tensor:
  return T * (T_max - T_min) + T_min

#@dataclass
#class Domain:
#    header: dict = field(default_factory=dict)
#    collocation: torch.Tensor = None
#    inital_points: torch.Tensor = None
#    inital_values: torch.Tensor = None
#    boundarie_keys: list[str] = field(default_factory=list)
#    boundarie_points: dict[str, torch.Tensor] = field(default_factory=dict)
#    boundarie_values: dict[str, torch.Tensor] = field(default_factory=dict)
#
#
#@dataclass
#class CData:
#  header:dict = field(default_factory=dict)
#  model:torch.nn.Module = None
#  domain:dict = field(default_factory=dict)
#  loss:torch.Tensor = None
#  optimizer:torch.optim.Optimizer = None

