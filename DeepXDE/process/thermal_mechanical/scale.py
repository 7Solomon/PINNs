import torch
class Scale:
    def __init__(self, domain_variables):
        self.x_min, self.x_max = domain_variables.spatial['x']
        self.y_min, self.y_max = domain_variables.spatial['y']
        self.t_min, self.t_max = domain_variables.temporal['t']

        self.Temperature = 10

        self.t = self.t_max - self.t_min
        #self.t = 1
        self.L = max(self.x_max - self.x_min, self.y_max - self.y_min)
        self.U = 0.1

    def sigma(self, E):
        return (E * self.U) / self.L  # [Pa]
#