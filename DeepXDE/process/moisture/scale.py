def scale_z(z):
    return z / 1.0
def rescale_z(z):
    return z * 1.0

def scale_t(t):
    return t / 4e6
def rescale_t(t):
    return t * 4e6

def scale_h(h):
    return h / 200
def rescale_h(h):
    return h * 200 

def scale_theta(theta):
    return theta / 0.5
def rescale_theta(theta):
    return theta * 0.5


class Scale:
    def __init__(self, domain_variables):
        self.z_min, self.z_max = list(domain_variables.spatial.values())[0]
        self.t_min, self.t_max = list(domain_variables.temporal.values())[0]

        self.T = (self.t_max - self.t_min)
        self.L = (self.z_max - self.z_min)

        self.H = 200

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