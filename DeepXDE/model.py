from utils.fourier_features import fourier_transform
import deepxde as dde
import numpy as np


def create_model(data, config, output_transform=None):
    fourier_transform_features = config.get('fourier_transform_features', None)
    decay = config.get('decay', None)
    if decay is not None:
        print(f'Using decay: {decay}')
   
    if fourier_transform_features:
        print(f'Using Fourier transform with {fourier_transform_features} features')
        pinn = dde.maps.FNN([config.input_dim*fourier_transform_features]+[50]*4+[config.output_dim], 'tanh', 'Glorot uniform')
        pinn.apply_feature_transform(lambda x: fourier_transform(x, num_features=fourier_transform_features, sigma=1.0))
    else:
        pinn = dde.maps.FNN([config.input_dim]+[50]*4+[config.output_dim], 'tanh', 'Glorot uniform')

    pinn.apply_output_transform(output_transform) if output_transform else None

    model = dde.Model(data, pinn)
    model.compile(**config.compile_args, loss_weights=config.loss_weights, decay=decay)
    return model
