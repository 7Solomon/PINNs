import datetime
import os
from utils import CData
import deepxde as dde
import numpy as np




def create_model(data):
    pinn = dde.maps.FNN([1]+[50]*3+[1], 'tanh', 'Glorot uniform')
    model = dde.Model(data, pinn)
    model.compile('adam', lr=1e-5)  # gestartet mit 1e-3
    return model
#def load_cData(model):
#    dir = os.listdir('models')
#    print('------------------------------')
#    print('Data:')
#    for i, file in enumerate(dir):
#        print(f'{i}: {file}')
#    print('------------------------------')
#    index = int(input('Welches Model laden?'))    
#    model.restore(f'models/{dir[index]}')
#    return model

