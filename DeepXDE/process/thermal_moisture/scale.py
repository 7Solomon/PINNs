from utils.metadata import BSaver
from material import concreteData
materialData = concreteData

class Scale(BSaver):
    def __init__(self, domain_variables):
        x_min, x_max = domain_variables.spatial['x']
        y_min, y_max = domain_variables.spatial['y']
        t_min, t_max = domain_variables.temporal['t']

        self.L = 0.2
        self.t = t_max - t_min
        
        self.Temperature = 30
        self.theta = 0.6

        self.c0 = 1.6e5  # J/m^3/K
        self.lamda = materialData.lamda_dry  # 
        print('GEHE SICER, dass materialData ist CONCRETE')

