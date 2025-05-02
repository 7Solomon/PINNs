from domain import *
from nn_stuff.model import create_model, load_model
from vis.test import plt_loss

if __name__ == "__main__":
    domain = get_domain()
    model, cData = create_model(domain)
    #model, cData = load_model()
    
    #print(cData.loss)
    plt_loss(cData.loss)
    