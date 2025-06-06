from dataclasses import asdict, dataclass, field, fields
import json
import math
import torch



@dataclass
class ConcreteData:
    cp: float = 8.5e3 # J/(kg*K)
    rho: float = 6e3  # kg/m^3
    k: float = 0.2 # W/(m*K)
    beta: float = 0.2  # 1/K

    E: float = 3e10 # Pa
    nu: float = 0.2
    thermal_expansion_coefficient: float = 1e-5  # 1/K
    
    # VG
    theta_r: float = 0.05                               # = 0.01
    theta_s: float = 0.35                               # = 0.12
    soil_water_retention: float = 2.0                   # = 0.4
    n: float = 2.0                                         # = 1.2
    m: float = 1.0 - 1.0/2.0                             #   1.2
    K_s: float = 1e-1#K_s: float = 10e-6  # RICHARDS für Concrete IS NE nen bisschen eine bitch also große K_s

    # Moisture (Darcy)
    K = 1e-9

    def alpha(self):
        return self.k/(self.rho*self.cp) # m^2/s
    


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


class SteadyHeatConfig(BConfig):
    input_dim: int = 2
    output_dim: int = 1


class TransientHeatConfig(BConfig):
    input_dim: int = 3
    output_dim: int = 1


class BernoulliBalkenConfig(BConfig):
    input_dim: int = 1
    output_dim: int = 1


class BernoulliBalken2DConfig(BConfig):
    input_dim: int = 2
    output_dim: int = 2
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0]

    def C(self, material):
        return material.E / ((1+material.nu)*(1-2*material.nu))  * torch.tensor([
                                                                                    [1-material.nu, material.nu, 0],
                                                                                    [material.nu, 1-material.nu, 0],
                                                                                    [0, 0, (1-2*material.nu)/2]
                                                                                ])

class BernoulliBalkenTconfig(BConfig):
    input_dim: int = 2
    output_dim: int = 1


    #def __post_init__(self):
    #    self.c = math.sqrt((self.E*self.I)/(self.rho*self.A))
    
    # load
    def f(self,x,t):
        #return self.E*self.I*(1.0-16*math.pi**2)*math.sin(x/math.pi)*math.cos((4*self.c*t)/math.pi)/self.L**3  # not 3d
        return (1.0-16.0*math.pi**2)*torch.sin(x)*torch.cos(4*math.pi*t) # scaled

class Richards1DConfig(BConfig):
    input_dim: int = 2
    output_dim: int = 1
    loss_weights: list[float] = [10.0, 1.0, 1.0, 1.0]

class RichardsMixed1DConfig(BConfig):
    input_dim: int = 2
    output_dim: int = 2
    loss_weights: list[float] = [1.0, 1.0, 10.0, 10.0, 10.0]


    # Van Genuchten params
    #theta_r: float = 0.01
    #theta_s: float = 0.12
    #alpha: float = 0.4
    #n: float = 1.2
    #m: float = 1.0 - 1.0/1.2
    #K_s: float = 1e-1#K_s: float = 10e-6  # RICHARDS für Concrete IS NE nen bisschen eine bitch also große K_s

    #def __post_init__(self): # macht probleme
    #    print('POST INIT')
    #    self.m = 1.0 - 1.0/self.n
class CooksMembranConfig(BConfig):
    input_dim: int = 2
    output_dim: int = 2

    # material
    E = 3e6  # Pa
    nu = 0.2
    

    def C(self):
        return (self.E/ (1-self.nu**2)) * torch.tensor([[1, self.nu, 0],
                                                        [self.nu, 1, 0],
                                                        [0, 0, (1-self.nu)/2]
                                                        ], dtype=torch.float64)
    
class Darcy2DConfig(BConfig):
    input_dim: int = 2
    output_dim: int = 1

class ThermalMechanical2DConfig(BConfig):
    input_dim: int = 3
    output_dim: int = 3
    loss_weights: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                           

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


concreteData = ConcreteData()