import deepxde as dde
import numpy as np


def create_model(data, config, output_transform=None):
    pinn = dde.maps.FNN([config.input_dim]+[50]*4+[config.output_dim], 'tanh', 'Glorot uniform')
    pinn.apply_output_transform(output_transform) if output_transform else None
    model = dde.Model(data, pinn)
    model.compile('adam', lr=1e-4)  # gestartet mit 1e-3
    return model
