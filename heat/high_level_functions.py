import argparse
import sys
from domain import generate_steady_domain
from vis.functions import visualize_field, visualize_loss
from nn_stuff.model import create_model, load_model


def manage_args(args: argparse.Namespace):
    if args.command == 'add':
        model, cData = create()
        if model and cData and args.vis:
            visualize_model(model, cData, args.vis)
        elif model:
            print('In Model generierung ist ein Fehler passiert', file=sys.stderr)

    elif args.command == 'load':
        model, cData = load_model_data()
        if model and args.vis:
            visualize_model(model, cData, args.vis)
        elif model is None:
            print('beim Laden ist ein Fehler passiert', file=sys.stderr)
def create():
    domain = generate_steady_domain()
    model, cData = create_model(domain)
    return model, cData
def load_model_data():
    try:
        model, cData = load_model()
        if not model or not cData.header or not cData.loss:
             raise ValueError('Model oder header oder loss not found.')
        return model, cData
    except FileNotFoundError:
        print('Error Datei nicht gefunden', file=sys.stderr)
        return None, None
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        return None, None

def visualize_model(model, cData, vis_type=None):
    if cData and vis_type:
        if vis_type == 'field' and cData.domain and cData.model:
            visualize_field(model, cData.domain)
        elif vis_type == 'loss' and cData.loss:
            visualize_loss(cData.loss)
        elif vis_type == 'all':
             if cData.domain and model:
                visualize_field(model, cData.domain)
             if cData.loss:
                visualize_loss(cData.loss)
