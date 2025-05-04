import os
from typing import Tuple
from moisture.nn_stuff.train import train_loop
import torch
import datetime
from nn_stuff.pinn import PINN

from utils import CData
from vars import device

#{'transient_heat': PINN}


def create_model(domain, conf, save_name=None) -> Tuple[PINN, CData]:

    model = PINN(conf.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)

    data = train_loop(model, optimizer, conf.mse_loss, domain, conf.epochs)
    model = data['model']
    Loss = data['loss']

    if not save_name:
        now = datetime.datetime.now()
        save_name = now.strftime("%Y-%m-%d_%H-%M-%S")

    save_data = CData(
        header={
            'name': save_name,
            'type': 'moisture',
            'domain': domain.header
        },
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        domain=domain,
        loss=Loss
    )
    if not os.path.exists(conf.model_path):
        os.makedirs(conf.model_path)

    torch.save(save_data, os.path.join(conf.model_path, f'{save_name}.pth'))
    return  model, save_data
  


def load_model_from_path(path):
    saved_cdata: CData = torch.load(path, map_location=device, weights_only=False)
    return saved_cdata

def load_model(conf) -> Tuple[PINN, CData]:
    if not os.path.exists(conf.model_path):
        os.makedirs(conf.model_path)

    for i, path in enumerate(os.listdir(conf.model_path)):
        print(f'{i}: {path}')
    print('----------')
    nr = int(input('Model nummer: '))

    loaded_cData = load_model_from_path(os.path.join(conf.model_path, os.listdir(conf.model_path)[nr]))
    model = PINN(conf.layers).to(conf.device)
    model.load_state_dict(loaded_cData.model)
    return model, loaded_cData
