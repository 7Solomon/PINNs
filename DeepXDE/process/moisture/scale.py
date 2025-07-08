from utils.metadata import BSaver
from material import concreteData
material_data = concreteData


class RichardsScale(BSaver):
    def __init__(self, domain_variables, **attributes):
        super().__init__(**attributes)
        self.z_min, self.z_max = domain_variables.spatial['z']
        self.t_min, self.t_max = domain_variables.temporal['t']
        self.L = (self.z_max - self.z_min)  # Length scale [m]
        self.T_domain = (self.t_max - self.t_min)
        self.T_physics = min(self.T_hydraulic, self.T_diffusion)

        # Determine time scale regime
        if self.T_domain >= self.T_physics:
            self.T = self.T_physics
            self.regime = "quasi-steady"
        else:
            self.T = self.T_domain
            self.regime = "transient"
        
        # Dimensionless numbers
        self.Da_hydraulic = self.T_domain / self.T_hydraulic
        self.Da_diffusion = self.T_domain / self.T_diffusion
        self.Pe = self.L * material_data.K_s / material_data.D_moisture

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

    def print_DEBUG(self):
        scale_name = self.__class__.__name__
        print(f"\n--- [{scale_name}] Analysis ---")
        if self.regime == "quasi-steady":
            print(f"Quasi-steady regime: using physics time scale")
        else:
            print(f"Transient regime: using domain time scale")

        print(f"\n=== TIME SCALE ===")
        print(f"Domain time:     {self.T_domain:.2e} s = {self.T_domain/86400:.1f} days")
        print(f"Hydraulic time:  {self.T_hydraulic:.2e} s = {self.T_hydraulic/86400:.1f} days")
        print(f"Diffusion time:  {self.T_diffusion:.2e} s = {self.T_diffusion/86400:.1f} days")
        print(f"Selected time:   {self.T:.2e} s = {self.T/86400:.1f} days")
        
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
        print("-" * (len(scale_name) + 12))

    


class HeadScale(RichardsScale):
    def __init__(self, domain_variables):
        super().__init__(domain_variables)
        self.H = self.h_char
    @property
    def value_scale_list(self):
        return [self.h_char]


class SaturationScale(RichardsScale):
    def __init__(self, domain_variables):
        super().__init__(domain_variables)
        self.theta = concreteData.theta_s - concreteData.theta_r
        self.H = self.h_char
    @property
    def value_scale_list(self):
        return [self.theta]


