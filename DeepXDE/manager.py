import argparse
from utils.variable_lr import VariableLearningRateCallback
from MAP import MAP
from utils.dynamic_loss import DynamicLossWeightCallback, SlowPdeLossWeightCallback
import vis
from utils.model_utils import load_function, save_function
from model import create_model
from utils.test import create_dde_data, heat_problem
import deepxde as dde
import matplotlib.pyplot as plt
from mpi4py import MPI

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
    test_parser.add_argument('--epochs', type=int, default=0, help='Training')
    test_parser.add_argument('--type', nargs='+', default=['heat', 'transient'],)



    return parser.parse_args()


def manage_args(args):
    comm = MPI.COMM_WORLD
    rank = comm.rank

    model, domain_vars, config, scale, loss_history = None, None, None, None, None

    process_type = args.type[0]
    subtype = args.type[1] if len(args.type) > 1 else None
    #if rank == 0:
    output_transform = MAP[process_type][subtype].get('output_transform', None)

    if args.command == 'add':
        domain_func = MAP[process_type][subtype]['domain']
        domain_vars = MAP[process_type][subtype]['domain_vars']
        config = MAP[process_type][subtype]['config']
        scale = MAP[process_type][subtype]['scale'](domain_vars)
    
        data = domain_func(domain_vars, scale)
        model = create_model(data, config, output_transform=None if not output_transform else lambda x,y: output_transform(x,y, scale))
    elif args.command == 'load':
        model, domain_vars, config, scale = load_function(process_type, subtype, output_transform=None if not output_transform else lambda x,y: output_transform(x,y, scale))
    elif args.command == 'list':
        raise NotImplementedError('List not implemented')
    elif args.command == 'test':
        #raise NotImplementedError('Test not implemented')
        #config = MAP['heat']['transient']['config']
        #domain_vars = MAP['heat']['transient']['domain_vars']
        #scale = MAP['heat']['transient']['scale'](domain_vars)
        #data = create_dde_data(heat_problem, {
        #    'num_domain': 1000,
        #    'num_boundary': 500,
        #    'num_initial': 200,
        #})
        #model = create_model(data, config, output_transform=None if not output_transform else lambda x,y: output_transform(x,y, scale))
        from process.heat.gnd import get_transient_fem
        from domain_vars import steady_heat_2d_domain, transient_heat_2d_domain
        #get_steady_fem(steady_heat_2d_domain)
        get_transient_fem(transient_heat_2d_domain)
        return
    else:
        raise ValueError('UNVALIDEr cOMmAND DU KEK')
        

    ### Train
    if hasattr(args, 'epochs') and args.epochs > 0:
        callbacks = []
        if hasattr(config, 'callbacks') and 'slowPdeAnnealing' in config.callbacks:
            callbacks.append(SlowPdeLossWeightCallback(pde_indices=config.pde_indices, final_weight=config.annealing_value))
        if hasattr(config, 'callbacks') and 'dynamicLossWeight' in config.callbacks:
            callbacks.append(DynamicLossWeightCallback())
        if hasattr(config, 'callbacks') and 'resample' in config.callbacks:
            callbacks.append(dde.callbacks.PDEPointResampler(period=1000))
        if hasattr(config, 'callbacks') and 'variable_lr_config' in config.callbacks:
            callbacks.append(VariableLearningRateCallback(**config.variable_lr_config))
        loss_history, train_state = model.train(iterations=args.epochs, callbacks=callbacks)
        #np.save('train_state.npy', train_state)
    else:
        loss_history = dde.model.LossHistory()

    ### VIS
    Vis = visualize(args.vis, process_type, subtype, model, loss_history, args, domain_vars, config, scale)

    ### SAVE

    #if rank == 0 and args.save:
    save_function(model, domain_vars, loss_history, config, scale, Vis, process_type, subtype)    



def visualize(vis_type, process_type, subtype, model, loss_history, args, domain_vars, config, scale):
    vis = {}
    if vis_type in ['loss', 'all']:
        try:
            loss_figures = MAP[process_type][subtype]['vis']['loss'](loss_history, config.loss_labels)
            vis.update(loss_figures)
        except KeyError as e:
            print(f"Warning: Loss visualization not available for {process_type}/{subtype}: {e}")
        except Exception as e:
            print(f"Error generating loss visualization: {e}")
    
    if vis_type in ['field', 'all']:
        #try:
            kwargs = MAP[process_type][subtype]['vis'].get('kwargs', {})
            field_figures = MAP[process_type][subtype]['vis']['field'](model, scale, **kwargs)
            vis.update(field_figures)
        #except KeyError as e:
        #    print(f"Warning: Field visualization not available for {process_type}/{subtype}: {e}")
        #except Exception as e:
        #    print(f"Error generating field visualization: {e}")
    
    #print(f"Generated visualizations: {list(vis.keys())}")
    return vis