from dataclasses import asdict, dataclass, field, fields
import json
import math
from utils.dynamic_loss import DynamicLossWeightCallback
import torch



@dataclass
class ConcreteData:
    cp: float = 8.5e3 # J/(kg*K)
    rho: float = 6e3  # kg/m^3
    k: float = 0.2 # W/(m*K)   # Vielliecht auch lamda, ja wajrsheinlich
    beta: float = 0.2  # 1/K

    E: float = 3e10 # Pa
    nu: float = 0.2
    thermal_expansion_coefficient: float = 1e-5  # 1/K
    
    # VG
    theta_r: float = 0.05                               # = 0.01
    theta_s: float = 0.35                               # = 0.12
    soil_water_retention: float = 14.5                 # = 0.4
    n: float = 1.89                                    # = 1.2
    m: float = 1.0 - 1.0/1.89                         #   1.2
    K_s: float = 1e-5#  K_s: float = 10e-6  # RICHARDS für Concrete IS NE nen bisschen eine bitch also große K_s

    # Moisture (Darcy) 
    #K: 1e-9    # 

    # Thermal Moisture
    rho_a: float = 1.29  # kg/m^3
    cp_a: float = 1e3  # J/(kg*K)

    rho_w:float = 1000  # kg/m^3
    cp_w: float = 4.2e3 

    phi: float = 0.9    # [%]
    L_v:float = 2.45e6  # J/kg    # Does change with Temp [(0,2.5),(25,2.45),(100,2.26)]
    D_v:float = 2e-5  # m^2/s

    lamda_dry: float = 1.4
    lamda_sat: float = 2.5

    def alpha(self):
        return self.k/(self.rho*self.cp) # m^2/s
    def C(self):  # 
        return self.E / ((1+self.nu)*(1-2*self.nu))  * torch.tensor([
                                                                        [1-self.nu, self.nu, 0],
                                                                        [self.nu, 1-self.nu, 0],
                                                                        [0, 0, (1-2*self.nu)/2]
                                                                    ])



@dataclass
class BConfig:
    @property
    def type(self) -> str:
        return self.__class__.__name__.lower()
    
    def to_json(self):
        """Convert the config to a JSON string."""
        data = asdict(self)
        data['__class__'] = self.__class__.__name__
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str):
        """Create a config instance from a JSON string."""
        data = json.loads(json_str)
        
        # Get the class name and remove it from data
        class_name = data.pop('__class__', cls.__name__)
        
        # Find the correct class in the global namespace
        config_class = globals().get(class_name, cls)
        
        # Filter data to only include fields that exist in the target class
        valid_fields = {f.name for f in fields(config_class)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        return config_class(**filtered_data)
    
    def get(self, attr_name, default=None):
        return getattr(self, attr_name, default)


###
#  Heat
###

class SteadyHeatConfig(BConfig):
    input_dim: int = 2
    output_dim: int = 1
    loss_weights: list[float] = [1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE', 'Left', 'Right']
    decay = None

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}


class TransientHeatConfig(BConfig):
    input_dim: int = 3
    output_dim: int = 1
    
    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE', 'Left', 'Right', 'Initial']



####
##  Moisture
###
class Richards1DConfig(BConfig):
    input_dim: int = 2
    output_dim: int = 1

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    #decay: list[str,int,float] = ['step', 2000, 0.5]
    loss_weights: list[float] = [100.0, 100.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE', 'Initial', 'Left', 'Right']
    callbacks: list[str] = ['resample']
    
    fourier_transform_features: int = 20
    #callbacks: list[str] = ['dynamicLossWeight']

class RichardsMixed1DConfig(BConfig):
    input_dim: int = 2
    output_dim: int = 2

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    loss_weights: list[float] = [1.0, 1.0, 10.0, 10.0, 10.0]
    loss_labels: list[str] = ['PDE_Head', 'PDE_Saturation', 'Initial', 'Left', 'Right']


class Darcy2DConfig(BConfig):
    input_dim: int = 2
    output_dim: int = 1

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    loss_weights: list[float] = [1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE', 'Left', 'Right']


###
#  Mechanical
##


class BernoulliBalkenConfig(BConfig):
    input_dim: int = 1
    output_dim: int = 1

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0]
    #loss_labels: list[str] = ['PDE_X', 'Left', 'Right']


class BernoulliBalken2DConfig(BConfig):
    input_dim: int = 2
    output_dim: int = 2

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    decay: list[str,int,float] = None
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE_X', 'PDE_Y', 'Left', 'Right']


class BernoulliBalkenTconfig(BConfig):
    input_dim: int = 2
    output_dim: int = 1

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


class CooksMembranConfig(BConfig):
    input_dim: int = 2
    output_dim: int = 2

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-3}
    #loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0]

class ThermalMechanical2DConfig(BConfig):
    input_dim: int = 3
    output_dim: int = 3

    compile_args: dict = {'optimizer': 'adam', 'lr': 1e-4}
    decay: list[str,int,float] = ['step', 10000, 0.5]
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    loss_labels: list[str] = ['PDE_X', 'PDE_Y', 'PDE_T', 'Left_T', 'Right_T', 'Bottom_U', 'Bottom_V', 'Initial_T']

class ThermalMoisture2DConfig(BConfig):
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


concreteData = ConcreteData()