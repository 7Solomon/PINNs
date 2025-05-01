import datetime
import os
from typing import List, Tuple
from utils import CData
from nn_stuff.temp_pinn import PINN, train_loop

import torch
import numpy as np
import torch.nn as nn
import time
from tqdm import tqdm

from vars import *

# Pinn stuff
def create_model(domain, save_name=None) -> Tuple[PINN, CData]:

    model = PINN(layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    data = train_loop(model, optimizer, mse_loss, domain, epochs=500)
    model = data['model']
    Loss = data['loss']

    if not save_name:
        now = datetime.datetime.now()
        save_name = now.strftime("%Y-%m-%d_%H-%M-%S")

    save_data = CData(
        header={
            'name': save_name,
            'domain': domain.header
        },
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        domain=domain,
        loss=Loss
    )
    torch.save(save_data, os.path.join(MODEL_PATH, f'{save_name}.pth'))
    return  model, save_data
  
    
def load_model_from_path(path):
    saved_cdata: CData = torch.load(path, map_location=device, weights_only=False)
    return saved_cdata
    #print(f'Loaded Model:  Domain: {header}')
    #return {
    #    'header': header,
    #    'model': model,
    #    'domain': domain,
    #    'loss': Loss
    #}

def load_model() -> Tuple[PINN, CData]:
    if os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH, exist_ok=True)

    for i, path in enumerate(os.listdir(MODEL_PATH)):
        print(f'{i}: {path}')
    print('----------')
    nr = int(input('Model nummer: '))

    loaded_cData = load_model_from_path(os.path.join(MODEL_PATH, os.listdir(MODEL_PATH)[nr]))
    model = PINN(layers).to(device)
    model.load_state_dict(loaded_cData.model)
                          
    return model, loaded_cData
