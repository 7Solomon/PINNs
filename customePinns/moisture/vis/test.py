import matplotlib.pyplot as plt
import numpy as np
from moisture.vars import *
from moisture.physics import S_e, WRC, HC

def plt_loss(loss):
    plt.plot(loss)
    plt.grid()
    plt.show()

def plot_saturation():
    h = torch.linspace(-2000, 50, steps=500)
    S_e_vals = S_e(h).detach().numpy()
    theta_vals = WRC(h).detach().numpy()
    K_vals = HC(h).detach().numpy()

    plt.figure(figsize=(10, 10))
    
    plt.subplot(1, 3, 1)
    plt.plot(h, S_e_vals, label='Saturation S_e', color='blue')
    plt.xlabel('Pressure head (h) [cm]')
    plt.ylabel('S_e')

    plt.subplot(1, 3, 2)
    plt.plot(h, theta_vals, label='Water retention Curve', color='green')
    plt.xlabel('Pressure head (h) [cm]')
    plt.ylabel('theta')

    plt.subplot(1, 3, 3)
    plt.plot(h, K_vals, label='Hydraulic Conductivity', color='red')
    plt.xlabel('Pressure head (h) [cm]')
    plt.ylabel('K')

    plt.tight_layout()
    plt.show()

