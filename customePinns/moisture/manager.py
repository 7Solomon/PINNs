from moisture.domain import get_domain
from moisture.nn_stuff.model import create_model
from moisture.vis.test import plot_saturation

def vis_functions():
    plot_saturation()

def create():
    domain = get_domain()
    model, cData = create_model(domain)
    return model, cData
