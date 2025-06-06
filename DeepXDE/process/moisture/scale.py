class Scale:
    def __init__(self, domain_variables):
        self.z_min, self.z_max = domain_variables.spatial['z']
        self.t_min, self.t_max = domain_variables.temporal['t']

        #self.T = (self.t_max - self.t_min)  # could lead to faulure
        self.T = 1
        self.L = (self.z_max - self.z_min)

        self.H = 10

        self.theta = 10

    #def scale_z(self, z):
    #    return (z - self.z_min) / (self.z_max - self.z_min)
#
    #def rescale_z(self, z):
    #    return z * (self.z_max - self.z_min) + self.z_min
#
    #def scale_t(self, t):
    #    return (t - self.t_min) / (self.t_max - self.t_min)
#
    #def rescale_t(self, t):
    #    return t * (self.t_max - self.t_min) + self.t_min
    #
    #def scale_h(self, h):
    #    return (h - 0) / (200 - 0)
    #def rescale_h(self, h):
    #    return h * (200 - 0) + 0


def scale_x(x):
    return x/ 1
def rescale_x(x):
    return x * 1