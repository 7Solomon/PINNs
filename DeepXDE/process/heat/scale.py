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
        
    def scale_x(self,x):
        return (x - self.x_min) / (self.x_max - self.x_min)
    def rescale_x(self, x):
        return x * (self.x_max - self.x_min) + self.x_min
    def scale_y(self, y):
        return (y - self.y_min) / (self.y_max - self.y_min)
    def rescale_y(self, y):
        return y * (self.y_max - self.y_min) + self.y_min
    def scale_t(self, t):
        return (t - self.t_min) / (self.t_max - self.t_min)
    def rescale_t(self, t):
        return t * (self.t_max - self.t_min) + self.t_min

    def scale_alpha_x(self, alpha):
        return alpha * (self.t_max / (self.x_max - self.x_min)**2)  # s_t/s_x²

    def scale_alpha_y(self, alpha):
        return alpha * (self.t_max / (self.y_max - self.y_min)**2)  # s_t/s_y²
    