import datetime
import json
import os
from utils.load_loss_history import load_loss_history_object
import torch
from dataclasses import dataclass, field
import numpy as np

import deepxde as dde

path_mapping = {
    'mechanic': {
        'fest_los':'process/models/mechanic/fest_los',
        'einspannung':'process/models/mechanic/einspannung',
        'fest_los_t':'process/models/mechanic/fest_los_t',
        'cooks':'process/models/mechanic/cooks_membrane',
        },
    'heat': {
        'steady': 'process/models/heat/steady',
        'transient': 'process/models/heat/transient',
        },
    'moisture': {
        '1d_head': 'process/models/moisture/1d_head',
        #'2d_head': 'process/models/moisture/2d_head',
        },
    }

#def load_loss_history(path):
#    if os.path.exists(path):
#        loss = np.load(path, allow_pickle=True)
#        return loss
#    else:
#        print(f'Loss nicht gefunden: {path}')
#        return None

#@dataclass
#class CData:
#  header:dict = field(default_factory=dict)
#  model:torch.nn.Module = None
#  #domain:dict = field(default_factory=dict)
#  loss:torch.Tensor = None

def save_cData(model, domain, Loss, type, subtype=None):
    local_MODEL_PATH = path_mapping[type]
    if subtype is not None:
        local_MODEL_PATH = local_MODEL_PATH[subtype] 
    elif isinstance(local_MODEL_PATH, dict):
        local_MODEL_PATH = local_MODEL_PATH[next(iter(local_MODEL_PATH.keys()))]

    now = datetime.datetime.now()
    save_name = now.strftime("%Y-%m-%d_%H-%M-%S")

    if not os.path.exists(local_MODEL_PATH):
        os.makedirs(local_MODEL_PATH)
    folder_name = os.path.join(local_MODEL_PATH, f'{save_name}')
    os.makedirs(folder_name, exist_ok=True)

    model.save(os.path.join(folder_name, 'model'))    
    #domain_params = {
    #    'num_domain': domain.num_domain,
    #    'num_boundary': domain.num_boundary
    #}
    
    # Save Loss history separately
    loss_path = os.path.join(folder_name,'loss.npy')
    dde.utils.external.save_loss_history(Loss, loss_path)
    
    #metadata = {
    #    'name': save_name,
    #    'type': 'deep_mechanic',
    #    'domain': domain_params,
    #    'model_path': model_path,
    #    'loss_path': loss_path
    #}
    
    #with open(os.path.join(local_MODEL_PATH, f'{save_name}_meta.json'), 'w') as f:
    #    import json
    #    json.dump(metadata, f)
    
def load_cData(model, type, subtype):
    local_MODEL_PATH = path_mapping[type][subtype] 
    os.makedirs(local_MODEL_PATH, exist_ok=True)
    dir = [d for d in os.listdir(local_MODEL_PATH) 
           if os.path.isdir(os.path.join(local_MODEL_PATH, d))]
    

    print('------------------------------')
    print('Data:')
    for i, file in enumerate(dir):
        print(f'{i}: {file}')
    print('------------------------------')
    
    index = int(input('Welches Model'))
    if index >= len(dir):
        print('INVALID')
        return None
    
    selected_folder = os.path.join(local_MODEL_PATH, dir[index])
    #model_path = os.path.join(selected_folder, 'model.pt')
    model_paths = [_ for _ in os.listdir(selected_folder) if _.endswith('.pt')]
    if len(model_paths) ==1:
        model_path = os.path.join(selected_folder, model_paths[0]) 
    else:
        print('Mehrere Modelle gefunden, bitte manuell auswählen')
        for i, file in enumerate(model_paths):
            print(f'{i}: {file}')
        index = int(input('Welches Model?'))
        if index >= len(model_paths):
            print('INVALID')
            return None
        model_path = os.path.join(selected_folder, model_paths[index])
    if not os.path.exists(model_path):
        print(f'Model nicht gefunden: {model_path}')
    
    model.restore(model_path)
    
    loss_path = os.path.join(selected_folder, 'loss.npy')
    loss = None
    if os.path.exists(loss_path):
        loss = load_loss_history_object(loss_path)
        #loss = dde.utils.external.load_loss_history(loss_path)
        #loss = np.load(loss_path, allow_pickle=True)
    else:
        print(f'Loss nicht gefunden: {loss_path}')
    
    
    return model, loss