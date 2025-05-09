from dataclasses import dataclass


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
  
steadyHeatConfig = SteadyHeatConfig()
bernoulliBalkenConfig = BernoulliBalkenConfig()
transientHeatConfig = TransientHeatConfig()