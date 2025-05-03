import argparse
import sys
from heat.domain import generate_steady_domain
from heat.nn_stuff.model import create_steady_model, load_steady_model


def create_steady():
    domain = generate_steady_domain()
    model, cData = create_steady_model(domain)
    return model, cData

def create_transient():
    raise NotImplementedError('Noch nicht implementiert, du kek')
    #domain = generate_transient_domain()
    model, cData = create_transient(domain)
    return model, cData


def load():
    try:
        model, cData = load_steady_model()
        if not model or not cData.header or not cData.loss:
             raise ValueError('Model oder header oder loss not found.')
        return model, cData
    except FileNotFoundError:
        print('Error Datei nicht gefunden', file=sys.stderr)
        return None, None
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        return None, None
