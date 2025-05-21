import argparse
import os
import vis
from utils.model_utils import load_cData, save_cData
from model import create_model
import process
import config

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


MAP = {
    'mechanic':{
        'fest_los': {
            'domain': process.mechanic.domain.get_fest_los_domain,
            'config': config.bernoulliBalkenConfig,
            'vis': {
                'loss': vis.plot_loss,
                'field': process.mechanic.vis.visualize_field,
                #'div': process.mechanic.vis.visualize_divergence,
            }
        },
       'einspannung': {
            'domain': process.mechanic.domain.get_einspannung_domain,
            'config': config.bernoulliBalkenConfig,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.mechanic.vis.visualize_field,
                #'div': process.mechanic.vis.visualize_divergence,
            }
        },
        'fest_los_t': {
            'domain': process.mechanic.domain.get_fest_los_t_domain,
            'config': config.bernoulliBalkenTConfig,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.mechanic.vis.visualize_field,
                #'div': process.mechanic.vis.visualize_divergence,
            }
        },
        'cooks': {
            'domain': process.mechanic.domain.get_cooks_domain,
            'config': config.cooksMembranConfig,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.mechanic.vis.visualize_field,
                #'div': process.mechanic.vis.visualize_divergence,
            }
        }
    },
    'heat':{
        'steady': {
            'domain': process.heat.domain.get_steady_domain,
            'config': config.steadyHeatConfig,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.heat.vis.visualize_steady_field,
                #'div': process.heat.vis.visualize_divergence,
            }
        },
        'transient': {
            'domain': process.heat.domain.get_transient_domain,
            'config': config.transientHeatConfig,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.heat.vis.visualize_steady_field,
                #'div': process.heat.vis.visualize_divergence,
            }
        }
    },
    'moisture':{
        '1d_head': {
            'domain': process.moisture.domain.get_1d_domain,
            'config': config.richards1DConfig,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.moisture.vis.visualize_field,
                #'div': process.moisture.vis.visualize_divergence,
            }
        },
    }
}

def manage_args(args):
    process_type = args.type[0]
    subtype = args.type[1] if len(args.type) > 1 else None
    
    
    if process_type not in MAP and not subtype in MAP[process_type]:
        raise ValueError(f'Du bist ein dummer Mensch, {process_type} ist nicht in der MAP')
    domain_func = MAP[process_type][subtype]['domain']
    config = MAP[process_type][subtype]['config']

    data = domain_func()


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
    
    if args.vis in MAP[process_type][subtype]['vis']:
        MAP[process_type][subtype]['vis'][args.vis](model)
    if args.vis == 'all':
        for vis in MAP[process_type][subtype]['vis']:
            MAP[process_type][subtype]['vis'][vis](model)

        


    #if args.vis in ['loss', 'all']:
    #    if loss_history is not None:
    #        from vis import plot_loss
    #        #dde.utils.external.plot_loss_history(loss_history)
    #        plot_loss(loss_history)
    #
    #if args.vis in ['field', 'all']:
    #    #dde.utils.external.plot_best_state(train_state)
    #    #plt.show()
    #    if process_type == 'mechanic':
    #        from process.mechanic.vis import visualize_field
    #        visualize_field(model, subtype, inverse_scale=None) 
    #    elif process_type == 'heat':
    #        from process.heat.vis import visualize_field
    #        from process.heat.scale import rescale_value
    #        visualize_field(model, subtype, inverse_scale=rescale_value)
    #    elif process_type == 'moisture':
    #        from process.moisture.vis import visualize_field
    #        visualize_field(model, subtype, inverse_scale=None)
    #
    #if args.vis in ['div', 'all']:
    #    if process_type == 'heat':
    #        from process.heat.vis import visualize_divergence
    #        from process.heat.scale import rescale_value
    #        visualize_divergence(model, subtype)
#
    
