import deepxde as dde
import numpy as np


def create_model(data, config):
    pinn = dde.maps.FNN([config.input_dim]+[50]*3+[config.output_dim], 'tanh', 'Glorot uniform')
    model = dde.Model(data, pinn)
    model.compile('adam', lr=1e-3)  # gestartet mit 1e-3
    return model
