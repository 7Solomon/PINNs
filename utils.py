from dataclasses import dataclass, field
import datetime
import os
import torch
import enum

class ConditionType(enum.Enum):
    DIRICHTLETT = 'D'
    NEUMANN = 'N'
    ROBIN = 'R'
    INITIAL = 'I'

@dataclass 
class Condition:
    key: str = None
    type: str = None
    points: torch.Tensor = None
    values: torch.Tensor = None

@dataclass
class Domain:
    header: dict = field(default_factory=dict)
    collocation: torch.Tensor = None
    initial_condition: Condition = None
    condition_keys: list[str] = field(default_factory=list)
    conditions: dict[str, Condition] = field(default_factory=dict)

@dataclass
class CData:
  header:dict = field(default_factory=dict)
  model:torch.nn.Module = None
  domain:dict = field(default_factory=dict)
  loss:torch.Tensor = None
  optimizer:torch.optim.Optimizer = None



def create_save_data(type, local_MODEL_PATH, model, optimizer, Loss, domain, save_name=None) -> CData:  
    if not save_name:
        now = datetime.datetime.now()
        save_name = now.strftime("%Y-%m-%d_%H-%M-%S")

    save_data = CData(
        header={
            'name': save_name,
            'type': type,
            'domain': domain.header
        },
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        domain=domain,
        loss=Loss
    )
    
    if not os.path.exists(local_MODEL_PATH):
        os.makedirs(local_MODEL_PATH)

    torch.save(save_data, os.path.join(local_MODEL_PATH, f'{save_name}.pth'))
    return save_data
