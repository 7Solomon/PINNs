from dataclasses import dataclass, field
import math
import torch


@dataclass
class ConcreteData:
    cp: float = 8.5e3 # J/(kg*K)
    rho: float = 6e3  # kg/m^3
    k: float = 0.2 # W/(m*K)

    
    def alpha(self):
        return self.k/(self.rho*self.cp) # m^2/s

@dataclass
class BConfig:
    @property
    def type(self) -> str:
        return self.__class__.__name__.lower()


@dataclass
class SteadyHeatConfig(BConfig):
    input_dim: int = 2
    output_dim: int = 1


@dataclass
class TransientHeatConfig(BConfig):
    input_dim: int = 3
    output_dim: int = 1


@dataclass
class BernoulliBalkenConfig(BConfig):
    input_dim: int = 1
    output_dim: int = 1


@dataclass
class BernoulliBalken2DConfig(BConfig):
    input_dim: int = 2
    #output_dim: int = 1

@dataclass
class BernoulliBalkenTconfig(BConfig):
    input_dim: int = 2
    output_dim: int = 1

    ## material
    #E = 10e10 # pa
    #I = 4e4 # m^4
    #rho = 2e3 #KG/m3
    #A = 1
    #c = field(default=None)
#
    ## geom
    #L = math.pi**2 # m
    #T = math.pi**2/200 # s
#
    #def __post_init__(self):
    #    self.c = math.sqrt((self.E*self.I)/(self.rho*self.A))
    
    # load
    def f(self,x,t):
        #return self.E*self.I*(1.0-16*math.pi**2)*math.sin(x/math.pi)*math.cos((4*self.c*t)/math.pi)/self.L**3  # not scaled
        return (1.0-16.0*math.pi**2)*torch.sin(x)*torch.cos(4*math.pi*t) # scaled

class RichardsConfig(BConfig):
    input_dim: int = 3
    output_dim: int = 1

    # Van Genuchten params
    theta_r: float = 0.0
    theta_s: float = 0.1
    alpha: float = 0.4
    n: float = 1.2
    m: float = 1.0 - 1.0/1.2
    K_s: float = 10e-6

    #def __post_init__(self): # macht probleme
    #    print('POST INIT')
    #    self.m = 1.0 - 1.0/self.n
class CooksMembranConfig(BConfig):
    input_dim: int = 2
    output_dim: int = 2

    # material
    E = 3e6  # Pa
    nu = 0.2
    C = None

    def __post_init__(self):
        self.C = (self.E/ (1-self.nu**2)) *torch.tensor([[1, self.nu, 0],
                                                        [self.nu, 1, 0],
                                                        [0, 0, (1-self.nu)/2]
                                                        ], dtype=torch.float32)
        

steadyHeatConfig = SteadyHeatConfig()
bernoulliBalkenConfig = BernoulliBalkenConfig()
bernoulliBalken2DConfig = BernoulliBalken2DConfig()
bernoulliBalkenTConfig = BernoulliBalkenTconfig()
cooksMembranConfig = CooksMembranConfig()
transientHeatConfig = TransientHeatConfig()
richardsConfig = RichardsConfig()


concreteData = ConcreteData()