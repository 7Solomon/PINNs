
from utils.metadata import BSaver
from material import concreteData
materialData = concreteData


class Scale(BSaver):
    def __init__(self, domain_variables):
        self.x_min, self.x_max = domain_variables.spatial['x']
        self.y_min, self.y_max = domain_variables.spatial['y']

        self.L = self.x_max - self.x_min
        self.H = self.y_max - self.y_min
    
    @property
    def sigma(self):
        return materialData.rho* materialData.g * self.L
    @property
    def U(self):
        return (materialData.rho * materialData.g * self.L**4) / (materialData.E * self.H**2) # From beam theory
