
from utils.metadata import BSaver
from material import concreteData
materialData = concreteData


class MechanicScale(BSaver):
    def __init__(self, domain_variables):
        self.x_min, self.x_max = domain_variables.spatial['x']
        self.y_min, self.y_max = domain_variables.spatial['y']

        self.L = self.x_max - self.x_min
        self.U_prescribed = 0.01 # [m]

        print(f"DEBUG: MechanicScale U: {self.U}")
        print(f"DEBUG: MechanicScale Sigma: {self.sigma}")

        #self.L = self.x_max - self.x_min
        #self.H = self.y_max - self.y_min
    
    @property
    def sigma(self):
        sigma_gravity = materialData.rho* materialData.g * self.L
        sigma_u_prescribed = materialData.E * self.U_prescribed / self.L
        return max(sigma_gravity, sigma_u_prescribed)
    @property
    def U(self):
        U_gravity = (materialData.rho * materialData.g * self.L**2) / materialData.E
        #return U_gravity
        return max(U_gravity, self.U_prescribed)
    @property
    def  value_scale_list(self):
        return [self.U, self.U]
    @property
    def input_scale_list(self):
        return [self.L, self.L]
    ## CONStrainst


class EnsemnbleMechanicScale(MechanicScale):
    def __init__(self, domain_variables):
        super().__init__(domain_variables)

    @property
    def e(self):
        return self.U / self.L    

    @property
    def value_scale_list(self):
        return [self.U, self.U, self.e, self.e, self.e, self.sigma, self.sigma, self.sigma]
    
    @property
    def input_scale_list(self):
        return [self.L, self.L]