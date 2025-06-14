
from utils.metadata import BSaver


class Scale(BSaver):
    def __init__(self, domain_variables):
        self.x_min, self.x_max = domain_variables.spatial['x']
        self.y_min, self.y_max = domain_variables.spatial['y']

        self.L = max(self.x_max - self.x_min, self.y_max - self.y_min)
        self.U = 1 # [L]
        self.f = self.U / self.L**2
    
    def sigma(self, E):
        return (E * self.U) / self.L
