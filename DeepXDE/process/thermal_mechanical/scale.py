import torch
class Scale:
    def __init__(self, domain_variables):
        self.x_min, self.x_max = domain_variables.spatial['x']
        self.y_min, self.y_max = domain_variables.spatial['y']
        self.t_min, self.t_max = domain_variables.temporal['t']

        self.Temperature = 1
        #self.Lx = self.x_max - self.x_min
        #self.Ly = self.y_max - self.y_min
        self.t = self.t_max - self.t_min
        self.L = max(self.x_max - self.x_min, self.y_max - self.y_min)

        #self.Ux = 1
        #self.Uy = 1
    def U(self, alpha):
        return alpha * self.L * self.Temperature


    def sigma_voigt(self, E, alpha):
        self.sigma_char = E * alpha * self.Temperature 
        return torch.tensor([
            self.sigma_char, 
            self.sigma_char, 
            self.sigma_char * 0.5
        ], dtype=torch.float64)
            