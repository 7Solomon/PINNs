import datetime
import os
from utils.metadata import Domain
from utils.directory import *
from utils.load_loss_history import load_loss_history_object
import torch
from dataclasses import dataclass, field
import numpy as np

import deepxde as dde



#def load_loss_history(path):


def save_function(model, domain: Domain, Loss, graphics:map, type:str, subtype:str):
    folder_name = get_folder_name(type, subtype)
    core = os.path.join(folder_name, 'core')
    os.makedirs(core, exist_ok=True)


    # MODEL
    model.save(os.path.join(core, 'model'))    

    # LOSS
    #loss_path = os.path.join(core,'loss.npy')
    #dde.utils.external.save_loss_history(Loss, loss_path)

    # DOMAIN
    domain_path = os.path.join(core, 'domain.json')
    save_dict_to_json(domain.to_dict(), domain_path)

    # GRAPHICS
    for key, graphic in graphics.items():
        graphic_path = os.path.join(folder_name, f'{key}.png')
        graphic.savefig(graphic_path)



def load_function(model, type:str, subtype:str):
    selected_folder = load_specific_process(type, subtype)
    core_folder = os.path.join(selected_folder, 'core')
    os.makedirs(core_folder, exist_ok=True)

    # MODEL
    model_path = filter_for_model_path(core_folder)
    model.restore(model_path)

    # LOSS
    #loss_path = os.path.join(core_folder, 'loss.npy')
    #loss = load_loss_history_object(loss_path)

    # DOMAIN
    domain_path = os.path.join(core_folder, 'domain.json')
    domain = Domain.from_dict(load_dict_from_json(domain_path))

    return model, None


    
