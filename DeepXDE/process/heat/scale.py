from utils.metadata import BSaver
from domain_vars import transient_heat_2d_domain
from domain_vars import steady_heat_2d_domain

from material import concreteData

class Scale(BSaver):
    def __init__(self, domain_variables):
        self.x_min, self.x_max = list(domain_variables.spatial.values())[0]
        self.y_min, self.y_max = list(domain_variables.spatial.values())[1]
        self.t_min, self.t_max = list(domain_variables.temporal.values())[0]

        self.L = max(self.x_max - self.x_min, self.y_max - self.y_min)
        self.t = self.t_max - self.t_min
        self.T = 100
        #self.L = 1
        #self.t = 1
        #self.T = 1
 
    @property
    def value_scale_list(self):
        return [self.T]
        #self.L = 1
        #self.t = 1
        #self.T = 1
        #print('GEHE SICER, dass materialData ist CONCRETE')
    @property
    def input_scale_list(self):
        return [self.L, self.L, self.t]
     
    #@property
    #def alpha(self):
    #    return (self.t / (self.L)**2)  # s_t/s_x²


        #self.Lx = 1
        #self.Ly = 1
        #self.t = 1
        #self.T = 1
        #self.alpha_x = 1
        #self.alpha_y = 1

