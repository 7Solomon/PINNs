import argparse
import os
from utils import load_cData, save_cData
from domain import get_domain
from model import create_model

import deepxde as dde

def parse_args():
    parser = argparse.ArgumentParser(description='DeepXDE model manager')
    subparsers = parser.add_subparsers(dest='command', help='Was wohl')
    
    # Add command parser
    add_parser = subparsers.add_parser('add', help='erstelle')
    add_parser.add_argument('--epochs', type=int, default=10000, help='Anzahll der Epochen')
    add_parser.add_argument('--save', action='store_true', help='Save?')
    add_parser.add_argument('--vis', choices=['loss', 'field', 'all'], default='all', 
                            help='Display options')
    
    # Load command parser
    load_parser = subparsers.add_parser('load', help='Load an existing model')
    load_parser.add_argument('--epochs', type=int, default=0, help='Training noch oben drauf')
    load_parser.add_argument('--save', action='store_true', help='Save?')
    load_parser.add_argument('--vis', choices=['loss', 'field', 'all'], default='all', 
                             help='Display options')
    
    return parser.parse_args()

def manage_args(args):
    data = get_domain()
    
    if args.command == 'add':
        model = create_model(data)
    elif args.command == 'load':
        model = create_model(data)
        model, cData = load_cData(model)
        loss_history = cData.loss
    else:
        raise ValueError('Nur load oder add')
    
    if hasattr(args, 'epochs') and args.epochs > 0:
        loss_history, train_state = model.train(epochs=args.epochs)
        #np.save('train_state.npy', train_state)
    
    if args.save:
        save_cData(model, data, loss_history)
    if args.vis in ['loss', 'all']:
        if loss_history is not None:
            from vis import plot_loss
            plot_loss(loss_history)
            #dde.saveplot(loss_history, issave=True, isshow=True)
    
    if args.vis in ['field', 'all']:
        from vis import visualize_field
        visualize_field(model)
