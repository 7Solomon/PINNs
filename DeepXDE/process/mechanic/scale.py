
from utils.metadata import BSaver
from material import concreteData
materialData = concreteData


class Scale(BSaver):
    def __init__(self, domain_variables):
        self.x_min, self.x_max = domain_variables.spatial['x']
        self.y_min, self.y_max = domain_variables.spatial['y']

        self.L = max(self.x_max - self.x_min, self.y_max - self.y_min)
    
    @property
    def sigma(self):
        return materialData.rho* materialData.g * self.L
    @property
    def U(self):
        return (self.sigma * self.L) / materialData.E
