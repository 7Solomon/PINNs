import os
import json
import datetime

from MAP import MAP


def get_model_path(type:str, subtype:str):
    local_MODEL_PATH = MAP[type][subtype]['path']
    os.makedirs(local_MODEL_PATH, exist_ok=True)
    return local_MODEL_PATH
def get_save_name():
    now = datetime.datetime.now()
    save_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    return save_name
def get_folder_name(type, subtype):
    local_MODEL_PATH = get_model_path(type, subtype)
    save_name = get_save_name()
    folder_name = os.path.join(local_MODEL_PATH, save_name)
    os.makedirs(folder_name, exist_ok=True)
    return folder_name
def save_dict_to_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
def load_dict_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def load_specific_process(type, subtype):
    # Get base path
    local_MODEL_PATH = get_model_path(type, subtype)
    dir = [d for d in os.listdir(local_MODEL_PATH) 
           if os.path.isdir(os.path.join(local_MODEL_PATH, d))]
    
    # PRINT
    print('------------------------------')
    print('Data:')
    for i, file in enumerate(dir):
        print(f'{i}: {file}')
    print('------------------------------')
    
    # SELECT
    index = int(input('Welches Model'))
    if index >= len(dir):
        print('INVALID')
        return None
    
    selected_folder = os.path.join(local_MODEL_PATH, dir[index])
    return selected_folder

def filter_for_model_path(core_folder):
    model_paths = [_ for _ in os.listdir(core_folder) if _.endswith('.pt')]
    if len(model_paths) ==1:
        model_path = os.path.join(core_folder, model_paths[0]) 
    else:
        print('Mehrere Modelle gefunden, bitte manuell auswählen')
        for i, file in enumerate(model_paths):
            print(f'{i}: {file}')
        index = int(input('Welches Model?'))
        if index >= len(model_paths):
            raise ValueError('INVALID')
        model_path = os.path.join(core_folder, model_paths[index])
    if not os.path.exists(model_path):
        ValueError(f'Model nicht gefunden: {model_path}')
    return model_path