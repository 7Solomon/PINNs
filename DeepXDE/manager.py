import argparse
import os
from utils.model_utils import load_cData, save_cData
from model import create_model

import deepxde as dde
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='DeepXDE model manager')
    subparsers = parser.add_subparsers(dest='command', help='Was wohl')
    
    # Add command parser
    add_parser = subparsers.add_parser('add', help='erstelle')
    add_parser.add_argument('--epochs', type=int, default=10000, help='Anzahll der Epochen')
    add_parser.add_argument('--type', nargs='+', default='mechanic',)
    add_parser.add_argument('--save', action='store_true', help='Save?')
    add_parser.add_argument('--vis', choices=['loss', 'field', 'div', 'all'], default='all', 
                            help='Display options')
    
    # Load command parser
    load_parser = subparsers.add_parser('load', help='Load an existing model')
    load_parser.add_argument('--type', nargs='+', default='mechanic',)
    load_parser.add_argument('--epochs', type=int, default=0, help='Training noch oben drauf')
    load_parser.add_argument('--save', action='store_true', help='Save?')
    load_parser.add_argument('--vis', choices=['loss', 'field', 'div', 'all'], default='all', 
                             help='Display options')
    
    test_parser = subparsers.add_parser('test', help='test stuff')
    test_parser.add_argument('--type', nargs='+', default='mechanic',)
    return parser.parse_args()

def manage_args(args):
    process_type = args.type[0]
    subtype = args.type[1] if len(args.type) > 1 else None
    
    
    if process_type == 'mechanic':
        from process.mechanic.domain import get_domain
        from config import bernoulliBalkenConfig, cooksMembranConfig, bernoulliBalkenTConfig
        if subtype == 'fest_los':
            config = bernoulliBalkenConfig
            pass
        elif subtype == 'einspannung':
            config = bernoulliBalkenConfig
            pass
        elif subtype == 'fest_los_t':
            config = bernoulliBalkenTConfig
        elif subtype == 'cooks':
            config = cooksMembranConfig
        else:
            config = bernoulliBalkenConfig
            subtype = 'fest_los'
        data = get_domain(subtype)

    elif process_type == 'heat':
        from config import steadyHeatConfig, transientHeatConfig
        if subtype == 'steady':
            config = steadyHeatConfig
        elif subtype == 'transient':
            config = transientHeatConfig
        else:
            config = steadyHeatConfig
            subtype = 'steady'
        from process.heat.domain import get_domain
        data = get_domain(subtype)
    
    elif process_type == 'moisture':
        from config import richardsConfig
        if subtype == '1d_head':
            config = richardsConfig
        else:
            config = richardsConfig
            subtype = '1d_head'
        from process.moisture.domain import get_domain
        data = get_domain(subtype)


    if args.command == 'add':
        model = create_model(data, config)
    elif args.command == 'load':
        model = create_model(data, config)
        model, loss_history = load_cData(model, process_type, subtype)

    elif args.command == 'list':
        raise NotImplementedError('List not implemented')

    else:
        raise ValueError('UNVALIDEr cOMmAND DU KEK')
    
    if hasattr(args, 'epochs') and args.epochs > 0:
        loss_history, train_state = model.train(epochs=args.epochs)
        #np.save('train_state.npy', train_state)
    
    if args.save:
        save_cData(model, data, loss_history, process_type, subtype=subtype)
    
    if args.vis in ['loss', 'all']:
        if loss_history is not None:
            from vis import plot_loss
            #dde.utils.external.plot_loss_history(loss_history)
            plot_loss(loss_history)
    
    if args.vis in ['field', 'all']:
        #dde.utils.external.plot_best_state(train_state)
        #plt.show()
        if process_type == 'mechanic':
            from process.mechanic.vis import visualize_field
            visualize_field(model, subtype, inverse_scale=None) 
        elif process_type == 'heat':
            from process.heat.vis import visualize_field
            from process.heat.scale import rescale_value
            visualize_field(model, subtype, inverse_scale=rescale_value)
        elif process_type == 'moisture':
            from process.moisture.vis import visualize_field
            visualize_field(model, subtype, inverse_scale=None)
    
    if args.vis in ['div', 'all']:
        if process_type == 'heat':
            from process.heat.vis import visualize_divergence
            from process.heat.scale import rescale_value
            visualize_divergence(model, subtype, inverse_scale=rescale_value)

    
