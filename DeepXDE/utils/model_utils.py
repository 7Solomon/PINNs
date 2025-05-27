import datetime
import os

from matplotlib import animation
from model import create_model
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
        graphic_path_base = os.path.join(folder_name, key)
        if isinstance(graphic, animation.Animation):
            graphic.save(f'{graphic_path_base}.gif', writer='ffmpeg', fps=30)
        else:
            graphic.savefig(f'{graphic_path_base}.png', dpi=300)



def load_function(type:str, subtype:str, config, output_transform=None):
    selected_folder = load_specific_process(type, subtype)

    core_folder = os.path.join(selected_folder, 'core')
    #os.makedirs(core_folder, exist_ok=True)

    # DOMAIN
    domain_path = os.path.join(core_folder, 'domain.json')
    domain = Domain.from_dict(load_dict_from_json(domain_path))


    # MODEL
    model_path = filter_for_model_path(core_folder)
    model = create_model(domain, config, output_transform=output_transform)
    model.restore(model_path)

    # LOSS
    #loss_path = os.path.join(core_folder, 'loss.npy')
    #loss = load_loss_history_object(loss_path)



    return model, domain


    
