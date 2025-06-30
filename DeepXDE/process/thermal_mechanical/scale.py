from utils.metadata import BSaver
import torch
from material import concreteData
materialData = concreteData
class Scale(BSaver):
    def __init__(self, domain_variables):
        self.x_min, self.x_max = domain_variables.spatial['x']
        self.y_min, self.y_max = domain_variables.spatial['y']
        self.t_min, self.t_max = domain_variables.temporal['t']

        self.Temperature = 10

        self.t = self.t_max - self.t_min
        self.L = min(self.x_max - self.x_min, self.y_max - self.y_min)
    @property
    def sigma(self):
        return materialData.E * materialData.thermal_expansion_coefficient * self.Temperature
    @property
    def U(self):
        return (self.sigma * self.L) / materialData.E
