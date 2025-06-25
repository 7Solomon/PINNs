from utils.metadata import BSaver
from material import concreteData
material_data = concreteData

class HeadScale(BSaver):
    def __init__(self, domain_variables):
        self.z_min, self.z_max = domain_variables.spatial['z']
        self.t_min, self.t_max = domain_variables.temporal['t']
        self.T = None

        self.L = (self.z_max - self.z_min)
        #self.T = 10e5
        #self.K = 1e-9
        #self.H = 7
        self.T_domain = (self.t_max - self.t_min)
        self.T_physics = min(self.T_hydraulic, self.T_diffusion, self.T_capillary)
        
        #
        if self.T_domain >= self.T_physics:
            self.T = self.T_physics
            self.regime = "quasi-steady"
            print(f"Quasi-steady regime: using physics time scale")
        else:
            self.T = self.T_domain
            self.regime = "transient"
            print(f"Transient regime: using domain time scale")


        ## FOr analysis
        self.Da_hydraulic = self.T_domain / self.T_hydraulic    # Damköhler number
        self.Da_diffusion = self.T_domain / self.T_diffusion
        self.Pe = self.L * material_data.K_s / material_data.D_moisture  # Peclet number

        self.print_DEBUG()

    @property
    def T_hydraulic(self):
        return self.L / material_data.K_s 
    @property
    def T_diffusion(self):
        return self.L**2 / material_data.D_moisture  # [s]
    @property
    def h_char(self):
        return 1.0 / material_data.alpha_vg  # [m]
    @property
    def T_capillary(self):
        return self.h_char / (material_data.K_s * material_data.alpha_vg)  # [s]

    def print_DEBUG(self):
        print(f"\n=== TIME SCALE ===")
        print(f"Domain time:     {self.T_domain:.2e} s = {self.T_domain/86400:.1f} days")
        print(f"Hydraulic time:  {self.T_hydraulic:.2e} s = {self.T_hydraulic/86400:.1f} days")
        print(f"Diffusion time:  {self.T_diffusion:.2e} s = {self.T_diffusion/86400:.1f} days")
        print(f"Capillary time:  {self.T_capillary:.2e} s = {self.T_capillary/86400:.1f} days")
        print(f"Selected time:   {self.T:.2e} s = {self.T/86400:.1f} days")
        print(f"Regime: {self.regime}")
        
        print(f"\n=== DIMENSIONLESS STUFF ===")
        print(f"Da_hydraulic = {self.Da_hydraulic:.2e} (domain_time/hydraulic_time)")
        print(f"Da_diffusion = {self.Da_diffusion:.2e} (domain_time/diffusion_time)")
        print(f"Peclet number = {self.Pe:.2e} (advection/diffusion)")
        
        if self.Da_hydraulic < 0.1:
            print("=> Hydraulic process is SLOW compared to domain time")
        elif self.Da_hydraulic > 10.0:
            print("=> Hydraulic process is FAST compared to domain time")
        else:
            print("=> Hydraulic and domain times are comparable")

        if self.Pe > 10.0:
            print("=> Advection-dominated flow")
        elif self.Pe < 0.1:
            print("=> Diffusion-dominated flow")
        else:
            print("=> Mixed advection-diffusion")

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
