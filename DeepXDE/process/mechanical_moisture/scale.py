from utils.metadata import BSaver
from material import concreteData

materialData = concreteData

class Scale(BSaver):
    def __init__(self, domain_variables):
        self.x_min, self.x_max = domain_variables.spatial['x']
        self.y_min, self.y_max = domain_variables.spatial['y']
        self.t_min, self.t_max = domain_variables.temporal['t']

        self.L = max(self.x_max - self.x_min, self.y_max - self.y_min)
        self.theta = 1
        #self.t = self.t_max - self.t_min
    @property
    def t(self):
        return self.L**2 / materialData.D_moisture
    @property
    def U(self):
        return self.L  # [epsilon*L]
    @property
    def sigma(self):
        return (materialData.E)
    @property
    def epsilon(self):
        return 1 # [sigma/E]
    @property
    def f(self):
        return self.sigma / (self.L*materialData.rho*materialData.g)
    
