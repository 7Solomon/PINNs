from dataclasses import dataclass, field
import math
import torch


@dataclass
class BConfig:
    @property
    def type(self) -> str:
        return self.__class__.__name__.lower()


@dataclass
class SteadyHeatConfig(BConfig):
    input_dim: int = 2

@dataclass
class TransientHeatConfig(BConfig):
    input_dim: int = 3
    alpha: float = 7.5e-7 

@dataclass
class BernoulliBalkenConfig(BConfig):
    input_dim: int = 1


@dataclass
class BernoulliBalken2DConfig(BConfig):
    input_dim: int = 2

@dataclass
class BernoulliBalkenTconfig(BConfig):
    input_dim: int = 2

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
    
steadyHeatConfig = SteadyHeatConfig()
bernoulliBalkenConfig = BernoulliBalkenConfig()
bernoulliBalken2DConfig = BernoulliBalken2DConfig()
bernoulliBalkenTConfig = BernoulliBalkenTconfig()
transientHeatConfig = TransientHeatConfig()
richardsConfig = RichardsConfig()