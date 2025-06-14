from utils.metadata import BSaver


class HeadScale(BSaver):
    def __init__(self, domain_variables):
        self.z_min, self.z_max = domain_variables.spatial['z']
        self.t_min, self.t_max = domain_variables.temporal['t']


        self.L = (self.z_max - self.z_min)
        self.T = 10e5
        self.K = 1e-9


        self.H = 7
class SaturationScale(BSaver):
    def __init__(self, domain_variables):
        self.z_min, self.z_max = domain_variables.spatial['z']
        self.t_min, self.t_max = domain_variables.temporal['t']

        self.L = (self.z_max - self.z_min)
        self.T = 10e5
        self.K = 5e-6
        self.S = 0.9

        self.theta = 0.01

        self.H = 7.0
