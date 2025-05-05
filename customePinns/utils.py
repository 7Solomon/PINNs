from dataclasses import dataclass, field
import datetime
import math
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
    scaled_points: torch.Tensor = None
    scaled_values: torch.Tensor = None


@dataclass
class Domain:
    header: dict = field(default_factory=dict)
    collocation: torch.Tensor = None
    initial_condition: Condition = None
    condition_keys: list[str] = field(default_factory=list)
    conditions: dict[str, Condition] = field(default_factory=dict)
    
    min_val: float = None
    max_val: float = None

    max_point: int = None
    min_point: int = None

    def scale_points(self):
        all_points = []
        points_to_scale = []

        # BCs
        for key in self.condition_keys:
            all_points.append(self.conditions[key].points)
            points_to_scale.append(self.conditions[key])

        # IC
        if self.initial_condition and self.initial_condition.points is not None:
            all_points.append(self.initial_condition.points)
            points_to_scale.append(self.initial_condition)

        if not all_points:
            raise ValueError('Hier fehler')
            

        # find min, max
        all_points_tensor = torch.cat(all_points)
        if all_points_tensor.numel() == 0:
            raise ValueError('Joe Moma')

        self.min_point = torch.min(all_points_tensor)
        self.max_point = torch.max(all_points_tensor)

        if self.max_point == self.min_point:
            raise ValueError('Max und min sind wieder gliceh was quatschig ist!')
        else:
            for condition in points_to_scale:
                condition.scaled_points = (condition.points - self.min_point) / (self.max_point - self.min_point)


    def scale_conditions(self):
        all_values = []
        conditions_to_scale = []

        # BCs
        for key in self.condition_keys:
            all_values.append(self.conditions[key].values)
            conditions_to_scale.append(self.conditions[key])

        # IC
        if self.initial_condition and self.initial_condition.values is not None:
            all_values.append(self.initial_condition.values)
            conditions_to_scale.append(self.initial_condition)

        if not all_values:
            raise ValueError('KA was passiert ist, aber ist leer!')
            

        # find min, max
        all_values_tensor = torch.cat(all_values)
        if all_values_tensor.numel() == 0:
            raise ValueError('Aj')

        self.min_val = torch.min(all_values_tensor)
        self.max_val = torch.max(all_values_tensor)

        if self.max_val == self.min_val:
            raise ValueError('Die max und min sind glich was quatschig ist!, führt zu div 0 error')
        else:
            for condition in conditions_to_scale:
                condition.scaled_values = (condition.values - self.min_val) / (self.max_val - self.min_val)
    def rescale_predictions(self, predictions : torch.Tensor) -> torch.Tensor:
        if self.min_val is None or self.max_val is None:
            raise ValueError('Min and max values are not set.')
        #print(f'predictions: {type(predictions)}')
        if isinstance(predictions, torch.Tensor):
            return predictions * (self.max_val - self.min_val) + self.min_val
        else:
            return [pred * (self.max_val - self.min_val) + self.min_val for pred in predictions]
    def rescale_points(self, points : torch.Tensor) -> torch.Tensor:
        if self.min_point is None or self.max_point is None:
            raise ValueError('Min and max values not set.')
        return points * (self.max_point - self.min_point) + self.min_point

        


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
