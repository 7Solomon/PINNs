from utils.metadata import BSaver
import torch
from material import concreteData
materialData = concreteData
class Scale(BSaver):
    def __init__(self, domain_variables):
        self.x_min, self.x_max = domain_variables.spatial['x']
        self.y_min, self.y_max = domain_variables.spatial['y']
        self.t_min, self.t_max = domain_variables.temporal['t']

        self.Temperature = 50

        #self.t = self.t_max - self.t_min
        self.L = max(self.x_max - self.x_min, self.y_max - self.y_min)
        self.t_phys = self.L**2 / materialData.alpha_thermal_diffusivity
        self.t_domain = self.t_max - self.t_min
        self.t = min(self.t_phys, self.t_domain)
        print(f"DEBUG: Scale t: {self.t}")
        print(f'sigma: {self.sigma}')
        print(f'U: {self.U}')

    @property
    def sigma(self):
        return materialData.E * materialData.thermal_expansion_coefficient * self.Temperature
    @property
    def U(self):
        return materialData.thermal_expansion_coefficient * self.Temperature * self.L

    @property
    def value_scale_list(self):
        return [self.U, self.U, self.Temperature]
    @property
    def input_scale_list(self):
        return [self.L, self.L, self.t]