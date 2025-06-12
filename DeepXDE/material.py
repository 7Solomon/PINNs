import torch

class CellularConcrete1:
    def __init__(self):
        # van Genuchten parameters
        self.theta_r = 0.0                      # Residual water content
        self.theta_s = 0.29                     # Saturated water content
        self.alpha = 7.16                       # [1/m]
        self.n = 7.45                           # Shape parameter
        self.m = 1.0 - 1.0 / self.n             # m = 1 - 1/n

        self.soil_water_retention = self.alpha  # just for me

        self.K_s = 1e-9                         # Saturated hydraulic conductivity [m/s]
        self.K = self.K_s                       # For 2D Darcy if constant

        # Porosity (matches theta_s if fully saturated)
        self.phi = 0.35

        # for thermal/moisture models
        self.heat_capacity = 850                # [J/kg·K]
        self.density = 600                      # [kg/m³]
        self.thermal_conductivity = 0.2         # [W/m·K]
        self.dynamic_viscosity = 0.2            # [arbitrary units, not used in Richards]




concreteData = CellularConcrete1()
