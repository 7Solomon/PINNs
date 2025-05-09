import datetime
import os
import torch
from dataclasses import dataclass, field
import numpy as np

local_MODEL_PATH = 'models'

@dataclass
class CData:
  header:dict = field(default_factory=dict)
  model:torch.nn.Module = None
  #domain:dict = field(default_factory=dict)
  loss:torch.Tensor = None
def save_cData(model, domain, Loss):
    now = datetime.datetime.now()
    save_name = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Don't try to save the domain object with unpicklable lambda functions
    # Instead, use DeepXDE's built-in save method for the model
    model_path = os.path.join(local_MODEL_PATH, f'{save_name}.ckpt')
    model.save(model_path)
    
    # Extract domain parameters if needed
    domain_params = {
        'num_domain': domain.num_domain,
        'num_boundary': domain.num_boundary
    }
    
    # Save Loss history separately
    loss_path = os.path.join(local_MODEL_PATH, f'{save_name}_loss.npy')
    if not os.path.exists(local_MODEL_PATH):
        os.makedirs(local_MODEL_PATH)
    
    np.save(loss_path, Loss)
    
    # Optional: Save metadata
    metadata = {
        'name': save_name,
        'type': 'deep_mechanic',
        'domain': domain_params,
        'model_path': model_path,
        'loss_path': loss_path
    }
    
    with open(os.path.join(local_MODEL_PATH, f'{save_name}_meta.json'), 'w') as f:
        import json
        json.dump(metadata, f)
    
    return model_path
def load_cData(model):
    dir = os.listdir(local_MODEL_PATH)
    print('------------------------------')
    print('Data:')
    # Look for DeepXDE checkpoint files instead
    ckpt_files = [file for file in dir if file.endswith('.ckpt')]
    for i, file in enumerate(ckpt_files):
        print(f'{i}: {file}')
    print('------------------------------')
    
    index = int(input('Which model to load? '))
    if index >= len(ckpt_files):
        print("Invalid index")
        return None
    
    model_path = os.path.join(local_MODEL_PATH, ckpt_files[index])
    
    # Use DeepXDE's restore method instead of torch.load
    model.restore(model_path)
    
    # Look for corresponding loss file
    base_name = ckpt_files[index].replace('.ckpt', '')
    loss_path = os.path.join(local_MODEL_PATH, f'{base_name}_loss.npy')
    loss = None
    if os.path.exists(loss_path):
        loss = np.load(loss_path)
    
    # Create a domain fresh (since we can't properly save/load it)
    from domain import get_domain
    domain = get_domain()
    
    return model, domain, loss