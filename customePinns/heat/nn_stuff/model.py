import datetime
import os
from typing import List, Tuple
from utils import *
from nn_stuff.pinn import PINN
from heat.nn_stuff.train import train_steady_loop, train_transient_loop

import torch
import numpy as np
import torch.nn as nn
import time
from tqdm import tqdm

from heat.vars import *
from vars import device

# Pinn stuff
def create_steady_model(domain, save_name=None) -> Tuple[PINN, CData]:
    model = PINN(layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    data = train_steady_loop(model, optimizer, mse_loss, domain, epochs=EPOCHS)
    save_data = create_save_data('steady_heat', MODEL_PATH, data['model'], optimizer, data['loss'], domain, save_name)

    return data['model'], save_data
def create_transient_model(domain, save_name=None) -> Tuple[PINN, CData]:
    model = PINN(layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    data = train_transient_loop(model, optimizer, mse_loss, domain, epochs=EPOCHS)
    save_data = create_save_data('transient_heat', MODEL_PATH, data['model'], optimizer, data['loss'], domain, save_name)

    return data['model'], save_data

  
def load_model_from_path(path):
    saved_cdata: CData = torch.load(path, map_location=device, weights_only=False)
    return saved_cdata

def load_steady_model() -> Tuple[PINN, CData]:
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    for i, path in enumerate(os.listdir(MODEL_PATH)):
        print(f'{i}: {path}')
    print('----------')
    nr = int(input('Model nummer: '))

    loaded_cData = load_model_from_path(os.path.join(MODEL_PATH, os.listdir(MODEL_PATH)[nr]))
    model = PINN(layers).to(device)
    model.load_state_dict(loaded_cData.model)
    return model, loaded_cData

def load_transient_model() -> Tuple[PINN, CData]:
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    for i, path in enumerate(os.listdir(MODEL_PATH)):
        print(f'{i}: {path}')
    print('----------')
    nr = int(input('Model nummer: '))

    loaded_cData = load_model_from_path(os.path.join(MODEL_PATH, os.listdir(MODEL_PATH)[nr]))
    model = PINN(layers).to(device)
    model.load_state_dict(loaded_cData.model)
    return model, loaded_cData

