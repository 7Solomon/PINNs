from config import cooksMembranConfig, bernoulliBalkenTConfig

class Scale:
    def __init__(self, domain_variables):
        self.x_min, self.x_max = domain_variables.spatial['x']
        self.y_min, self.y_max = domain_variables.spatial['y']

        self.L = max(self.x_max - self.x_min, self.y_max - self.y_min)
        self.U = 1 # [L]
        self.f = self.U / self.L**2

    def sigma(self, E):
        return (E * self.U) / self.L

    #def Ux(self, alpha):
    #    return alpha * self.L
    #def Uy(self, alpha):
    #    return alpha * self.L

    #def scale_u(u):
    #    return u/bernoulliBalkenTConfig.L
    #def scale_x(x):
    #    return x/bernoulliBalkenTConfig.L
    #def scale_t(t):
    #    return (bernoulliBalkenTConfig.c*t)/(bernoulliBalkenTConfig.L**2)
    #def scale_f(f):
    #    return (f*bernoulliBalkenTConfig.L**3)/(bernoulliBalkenTConfig.E*bernoulliBalkenTConfig.I)