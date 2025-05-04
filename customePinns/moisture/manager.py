from config import MoistureConfig

def vis_functions():
    from moisture.vis.test import plot_saturation
    plot_saturation()

def create():
    from moisture.domain import get_domain
    from nn_stuff.model import create_model
    domain = get_domain()
    model, cData = create_model(domain, MoistureConfig())
    return model, cData

def load():
    from nn_stuff.model import load_model
    model, cData = load_model(MoistureConfig())
    return model, cData
