from dataclasses import dataclass
import torch

@dataclass
class Material:
    # Thermal Moisture
    rho_a: float = 1.225
    cp_a: float = 1005.0
    rho_w: float = 998.0
    cp_w: float = 4182.0

    # Base
    g:float = 9.81  # [m/s^2]  

#class CellularConcrete1:
#    def __init__(self):
#        # van Genuchten parameters
#        self.theta_r = 0.0                      # Residual water content
#        self.theta_s = 0.29                     # Saturated water content
#        self.alpha = 7.16                       # [1/m]
#        self.n = 7.45                           # Shape parameter
#        self.m = 1.0 - 1.0 / self.n             # m = 1 - 1/n
#
#        self.soil_water_retention = self.alpha  # just for me
#
#        self.K_s = 1e-9                         # Saturated hydraulic conductivity [m/s]
#        self.K = self.K_s                       # For 2D Darcy if constant
#
#        # Porosity (matches theta_s if fully saturated)
#        self.phi = 0.35
#
#        # for thermal/moisture models
#        self.heat_capacity = 850                # [J/kg·K]
#        self.density = 600                      # [kg/m³]
#        self.thermal_conductivity = 0.2         # [W/m·K]
#        self.dynamic_viscosity = 0.2            # [arbitrary units, not used in Richards]


class ConcreteData(Material):
    # Thermal
    cp: float = 8.5e2 # [J/(kg*K)]
    rho: float = 2.4e3  # [kg/m^3]
    k: float = 1.4 # [W/(m*K)]   # Vielliecht auch lamda, ja wajrsheinlich
    beta_thermal_stress: float = 0.2  # [1/K]  # auch manchmal deriveved
    # Mechanical
    E: float = 3e9 # [Pa]
    nu: float = 0.2
    thermal_expansion_coefficient: float = 1.2e-5  # [1/K]

    # VG
    theta_r: float = 0.02                      
    theta_s: float = 0.15   
    alpha_vg: float = 0.5  # [1/m]
    n_vg: float = 1.5                         
    K_s: float = 1e-9 # [m/s]

    phi: float = 0.15    # [%]  # Needs to match theta_s if fully saturated
    L_v:float = 2.45e6  # J/kg    # Does change with Temp [(0,2.5),(25,2.45),(100,2.26)]
    D_v:float = 2.6e-5  # m^2/s

    lamda_dry: float = 1.0  # [W/(m*K)]
    lamda_sat: float = 1.8  # [W/(m*K)]

    # Moisture Mechanical
    alpha_biot: float = 0.8  # Biot coefficient for moisture coupling
    D_moisture: float = 5e-10  # [m^2/s]  # Moisture diffusivity
    strain_moisture_coulling_coef: float = 1.0 #NON STANDART, but needed for coupling, describes how much strain influences moisture diffusivity 
    
    @property
    def m_vg(self) -> float:
        return 1.0 - 1.0 / self.n_vg if self.n_vg != 0 else 0.0
    @property
    def Sr(self) -> float:
        return self.theta_r / self.theta_s if self.theta_s > 0 else 0.0
    @property
    def alpha_thermal_diffusivity(self) -> float:
        return self.k / (self.rho * self.cp)

    def C_stiffness_matrix(self):  
        factor = self.E / ((1 + self.nu) * (1 - 2 * self.nu))
        
        C11 = factor * (1 - self.nu)
        C12 = factor * self.nu
        C33 = self.E / (2 * (1 + self.nu)) # Plane stress assumption
        
        return torch.tensor([
            [C11, C12, 0],
            [C12, C11, 0],
            [0, 0, C33]
        ])


class SandData(Material):
    # Thermal
    cp: float = 800.0 
    rho: float = 1600.0
    k: float = 0.6 
    beta_thermal_stress: float = 0.1 
    
    # Mechanical 
    E: float = 50e6
    nu: float = 0.3 
    thermal_expansion_coefficient: float = 1.0e-5  

    # Van Genuchten
    theta_r: float = 0.
    theta_s: float = 0.43 
    alpha_vg: float = 14.5  # [1/m] 
    n_vg: float = 2.68     
    K_s: float = 2.9e-5 

    phi: float = 0.43  # Porosity [%] (Should be consistent with theta_s)
    L_v: float = 2.45e6
    D_v: float = 2.6e-5

    lambda_dry: float = 0.3   # [W/(m*K)]
    lambda_sat: float = 2.5   # [W/(m*K)] (Can be higher, up to 4 for quartz sand)

    @property
    def m_vg(self) -> float:
        return 1.0 - 1.0 / self.n_vg if self.n_vg != 0 else 0.0
    @property
    def Sr(self) -> float:
        return self.theta_r / self.theta_s if self.theta_s > 0 else 0.0
    @property
    def alpha_thermal_diffusivity(self) -> float:
        return self.k / (self.rho * self.cp)

    def C_stiffness_matrix(self):
        # Assuming isotropic linear elasticity, plane stress
        factor = self.E / ((1 + self.nu) * (1 - 2 * self.nu))
        C11 = factor * (1 - self.nu)
        C12 = factor * self.nu
        C33 = factor * (1 - 2 * self.nu) / 2
        
        return torch.tensor([
            [C11, C12, 0],
            [C12, C11, 0],
            [0, 0, C33]
        ])



concreteData = ConcreteData()
sandData = SandData()
