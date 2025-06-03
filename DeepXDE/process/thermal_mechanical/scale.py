import torch
class Scale:
    def __init__(self, domain_variables):
        self.x_min, self.x_max = list(domain_variables.spatial.values())[0]
        self.y_min, self.y_max = list(domain_variables.spatial.values())[1]
        self.t_min, self.t_max = list(domain_variables.temporal.values())[0]

        self.Temperature = 100
        self.Lx = self.x_max - self.x_min
        self.Ly = self.y_max - self.y_min
        self.t = self.t_max - self.t_min

        self.Ux = 1
        self.Uy = 1

    def sigma_voigt(self, E, alpha):
        self.Ux = alpha * self.Lx * self.Temperature
        self.Uy = alpha * self.Ly * self.Temperature

        self.sigma_char = E * alpha * self.Temperature 
        return torch.tensor([
            self.sigma_char, 
            self.sigma_char, 
            self.sigma_char * 0.5
        ], dtype=torch.float64)
            