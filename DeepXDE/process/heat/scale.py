from domain_vars import transient_heat_2d_domain
from domain_vars import steady_heat_2d_domain

MIN = 0.0
MAX = 100.0


def scale_value(input):
    return (input - MIN) / (MAX - MIN)
def rescale_value(input):
    return input * (MAX - MIN) + MIN


class Scale:
    def __init__(self, domain_variables):
        self.x_min, self.x_max = list(domain_variables.spatial.values())[0]
        self.y_min, self.y_max = list(domain_variables.spatial.values())[1]
        self.t_min, self.t_max = list(domain_variables.temporal.values())[0]

        self.Lx = self.x_max - self.x_min
        self.Ly = self.y_max - self.y_min
        self.t = self.t_max - self.t_min
        self.T = 100

        self.alpha_x = (self.t / (self.x_max - self.x_min)**2)  # s_t/s_x²
        self.alpha_y = (self.t / (self.y_max - self.y_min)**2)  # s_t/s_y²

        #self.Lx = 1
        #self.Ly = 1
        #self.t = 1
        #self.T = 1
        #self.alpha_x = 1
        #self.alpha_y = 1