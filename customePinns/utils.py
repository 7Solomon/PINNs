from dataclasses import dataclass, field
import datetime
import math
import os
from typing import Dict, List, Optional, Tuple, Union
import torch
import enum

class ConditionType(enum.Enum):
    DIRICHLET = 'D'
    NEUMANN = 'N'
    ROBIN = 'R'
    INITIAL = 'I'

@dataclass 
class Condition:
    key: str = None
    type: str = None
    points: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None

    scaled_points: Optional[torch.Tensor] = None
    scaled_values: Optional[torch.Tensor] = None

@dataclass
class DomainV2:
    dimensions: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    is_time_dependent: bool = False
    time_dim: Optional[str] = None
    boundary_conditions: List[Condition] = field(default_factory=list)
    initial_conditions: List[Condition] = field(default_factory=list)
    
    collocation_points: Optional[torch.Tensor] = None
    scaled_collocation_points: Optional[torch.Tensor] = None
    boundary_points: Dict[str, torch.Tensor] = field(default_factory=dict)
    initial_points: Optional[torch.Tensor] = None
    
    # Scaling parameters - per dimension for points
    point_scales: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Scaling for values (solution)
    value_scale: Dict[str, float] = field(default_factory=dict)
    
    @property
    def dim_names(self) -> List[str]:
        return list(self.dimensions.keys())
    
    @property
    def spatial_dims(self) -> List[str]:
        if self.time_dim:
            return [dim for dim in self.dim_names if dim != self.time_dim]
        return self.dim_names
    
    @property
    def ndims(self) -> int:
        return len(self.dimensions)
    
    def scale(self, auto_scale: bool = True):
        """Scale both points and values in a single method call"""
        if auto_scale:
            self.compute_scaling_factors()
        self.apply_point_scaling()
        self.apply_value_scaling()
    
    def compute_scaling_factors(self):
        """Compute scaling factors for both points and values"""
        # 1. Compute scaling factors for points (per dimension)
        all_points = []
        
        # Gather all boundary points
        for face, points in self.boundary_points.items():
            if points is not None and points.numel() > 0:
                all_points.append(points)
        
        # Add collocation points
        if self.collocation_points is not None and self.collocation_points.numel() > 0:
            all_points.append(self.collocation_points)
        
        # Add initial points (for time-dependent problems)
        if self.initial_points is not None and self.initial_points.numel() > 0:
            all_points.append(self.initial_points)
        
        if not all_points:
            raise ValueError("No points available for scaling computation")
        
        # Concatenate all points and compute min/max per dimension
        all_points_tensor = torch.cat(all_points, dim=0)
        
        for i, dim_name in enumerate(self.dim_names):
            dim_min = torch.min(all_points_tensor[:, i]).item()
            dim_max = torch.max(all_points_tensor[:, i]).item()
            
            if dim_min == dim_max:
                raise ValueError(f"Dimension {dim_name} has identical min and max values ({dim_min})")
            
            self.point_scales[dim_name] = {
                'min': dim_min,
                'max': dim_max,
                'range': dim_max - dim_min
            }
        
        # 2. Compute scaling factors for values
        all_values = []
        
        # Gather boundary condition values
        for condition in self.boundary_conditions:
            if hasattr(condition, 'values') and condition.values is not None:
                all_values.append(condition.values)
        
        # Gather initial condition values
        for condition in self.initial_conditions:
            if hasattr(condition, 'values') and condition.values is not None:
                all_values.append(condition.values)
        
        if all_values:
            all_values_tensor = torch.cat(all_values, dim=0)
            value_min = torch.min(all_values_tensor).item()
            value_max = torch.max(all_values_tensor).item()
            
            if value_min == value_max:
                raise ValueError(f"Values have identical min and max ({value_min}), cannot scale")
            
            self.value_scale = {
                'min': value_min,
                'max': value_max,
                'range': value_max - value_min
            }
    
    def apply_point_scaling(self):
        """Apply computed scaling to all points"""
        # 1. Scale boundary points
        for face, points in self.boundary_points.items():
            if points is not None and points.numel() > 0:
                scaled_points = torch.clone(points)
                for i, dim_name in enumerate(self.dim_names):
                    scale = self.point_scales[dim_name]
                    scaled_points[:, i] = (points[:, i] - scale['min']) / scale['range']
                self.boundary_points[f"{face}_scaled"] = scaled_points
        
        # 2. Scale collocation points
        if self.collocation_points is not None and self.collocation_points.numel() > 0:
            scaled_points = torch.clone(self.collocation_points)
            for i, dim_name in enumerate(self.dim_names):
                scale = self.point_scales[dim_name]
                scaled_points[:, i] = (self.collocation_points[:, i] - scale['min']) / scale['range']
            self.scaled_collocation_points = scaled_points
        
        # 3. Scale initial points
        if self.initial_points is not None and self.initial_points.numel() > 0:
            scaled_points = torch.clone(self.initial_points)
            for i, dim_name in enumerate(self.dim_names):
                scale = self.point_scales[dim_name]
                scaled_points[:, i] = (self.initial_points[:, i] - scale['min']) / scale['range']
            self.initial_points_scaled = scaled_points
    
    def apply_value_scaling(self):
        """Apply computed scaling to all condition values"""
        if not self.value_scale:
            return
            
        # Scale boundary condition values
        for condition in self.boundary_conditions:
            if hasattr(condition, 'values') and condition.values is not None:
                condition.scaled_values = (condition.values - self.value_scale['min']) / self.value_scale['range']
        
        # Scale initial condition values
        for condition in self.initial_conditions:
            if hasattr(condition, 'values') and condition.values is not None:
                condition.scaled_values = (condition.values - self.value_scale['min']) / self.value_scale['range']
    
    def scale_tensor(self, tensor: torch.Tensor, dimensions: bool = True) -> torch.Tensor:
        """Scale an arbitrary tensor of points or values"""
        if dimensions:
            # Scale points (based on dimensions)
            if not self.point_scales:
                raise ValueError("Point scaling factors not computed, call compute_scaling_factors() first")
                
            result = torch.clone(tensor)
            for i, dim_name in enumerate(self.dim_names):
                if i < tensor.shape[1]:  # Only scale dimensions that exist in tensor
                    scale = self.point_scales[dim_name]
                    result[:, i] = (tensor[:, i] - scale['min']) / scale['range']
            return result
        else:
            # Scale values
            if not self.value_scale:
                raise ValueError("Value scaling factors not computed, call compute_scaling_factors() first")
                
            return (tensor - self.value_scale['min']) / self.value_scale['range']
    
    def unscale_tensor(self, tensor: torch.Tensor, dimensions: bool = True) -> torch.Tensor:
        """Convert scaled tensor back to original range"""
        if dimensions:
            # Unscale points
            if not self.point_scales:
                raise ValueError("Point scaling factors not computed")
                
            result = torch.clone(tensor)
            for i, dim_name in enumerate(self.dim_names):
                if i < tensor.shape[1]:  # Only unscale dimensions that exist in tensor
                    scale = self.point_scales[dim_name]
                    result[:, i] = tensor[:, i] * scale['range'] + scale['min']
            return result
        else:
            # Unscale values
            if not self.value_scale:
                raise ValueError("Value scaling factors not computed")
                
            return tensor * self.value_scale['range'] + self.value_scale['min']
    
    def rescale_predictions(self, predictions: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Rescale network predictions back to original value range"""
        if not self.value_scale:
            raise ValueError("Value scaling not set. Call compute_scaling_factors() first.")
            
        if isinstance(predictions, torch.Tensor):
            return predictions * self.value_scale['range'] + self.value_scale['min']
        else:
            return [pred * self.value_scale['range'] + self.value_scale['min'] for pred in predictions]

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
