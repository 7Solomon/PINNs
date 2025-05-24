import argparse
from MAP import MAP
import vis
from utils.model_utils import load_function, save_function
from model import create_model

import deepxde as dde
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='DeepXDE model manager')
    subparsers = parser.add_subparsers(dest='command', help='Was wohl')
    
    # Add command parser
    add_parser = subparsers.add_parser('add', help='erstelle')
    add_parser.add_argument('--epochs', type=int, default=10000, help='Anzahll der Epochen')
    add_parser.add_argument('--type', nargs='+', default=['mechanic', 'fest_los'],)
    add_parser.add_argument('--save', action='store_true', help='Save?')
    add_parser.add_argument('--vis', choices=['loss', 'field', 'div', 'all'], default='all', 
                            help='Display options')
    
    # Load command parser
    load_parser = subparsers.add_parser('load', help='Load an existing model')
    load_parser.add_argument('--type', nargs='+', default=['mechanic', 'fest_los'],)
    load_parser.add_argument('--epochs', type=int, default=0, help='Training noch oben drauf')
    load_parser.add_argument('--save', action='store_true', help='Save?')
    load_parser.add_argument('--vis', choices=['loss', 'field', 'div', 'all'], default='all', 
                             help='Display options')
    
    test_parser = subparsers.add_parser('test', help='test stuff')
    test_parser.add_argument('--type', nargs='+', default=['mechanic', 'fest_los'])
    return parser.parse_args()


def manage_args(args):
    process_type = args.type[0]
    subtype = args.type[1] if len(args.type) > 1 else None
    if process_type not in MAP and not subtype in MAP[process_type]:
        raise ValueError(f'Du bist ein dummer Mensch, {process_type} ist nicht in der MAP')
    
    ### GET FROM MAP
    domain_func = MAP[process_type][subtype]['domain']
    config = MAP[process_type][subtype]['config']

    data, domain_vars = domain_func()


    if args.command == 'add':
        model = create_model(data, config)
    elif args.command == 'load':
        model = create_model(data, config)
        model, loss_history = load_function(model, process_type, subtype)
    elif args.command == 'list':
        raise NotImplementedError('List not implemented')
    else:
        raise ValueError('UNVALIDEr cOMmAND DU KEK')
    

    ### Train
    if hasattr(args, 'epochs') and args.epochs > 0:
        loss_history, train_state = model.train(iterations=args.epochs)
        #np.save('train_state.npy', train_state)

    ### VIS
    Vis = visualize(args.vis, process_type, subtype, model, loss_history, args)            
    
    ### SAVE
    if args.save:
        save_function(model, domain_vars, loss_history, Vis, process_type, subtype)    



def visualize(vis_type, process_type, subtype, model, loss_history, args):
    vis = {}
    if vis_type in ['loss', 'all']:
        try:
            loss_figures = MAP[process_type][subtype]['vis']['loss'](loss_history)
            vis.update(loss_figures)
        except KeyError as e:
            print(f"Warning: Loss visualization not available for {process_type}/{subtype}: {e}")
        except Exception as e:
            print(f"Error generating loss visualization: {e}")
    
    if vis_type in ['field', 'all']:
        try:
            field_figures = MAP[process_type][subtype]['vis']['field'](model, subtype)
            vis.update(field_figures)
        except KeyError as e:
            print(f"Warning: Field visualization not available for {process_type}/{subtype}: {e}")
        except Exception as e:
            print(f"Error generating field visualization: {e}")
    
    print(f"Generated visualizations: {list(vis.keys())}")
    return vis