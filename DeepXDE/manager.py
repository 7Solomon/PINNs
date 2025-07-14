import argparse
from vis import plot_mse
from points import get_evaluation_points
from MAP import MAP
from utils.model_utils import load_function, save_function
from model import create_flexible_model, get_callbacks
import deepxde as dde
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
    add_parser.add_argument('--comment', type=str, default='', help='Comment that gets saved with the model')
    
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
    process_type = args.type[0]
    subtype = args.type[1] if len(args.type) > 1 else 'default'
    model, domain_vars, config, scale, loss_history = None, None, None, None, None


    domain_func = MAP[process_type][subtype]['domain']
    domain_vars = MAP[process_type][subtype]['domain_vars']
    gnd_func = MAP[process_type][subtype]['gnd_function']

    point_data = get_evaluation_points(domain_vars)
    fem_value_points = gnd_func(domain_vars, point_data, comm)

    #if rank == 0:
    output_transform = MAP[process_type][subtype].get('output_transform', None)

    if args.command == 'add':
        config = MAP[process_type][subtype]['config']
        scale = MAP[process_type][subtype]['scale'](domain_vars)
    
        data = domain_func(domain_vars, scale)
        model = create_flexible_model(data, config, output_transform=None if not output_transform else lambda x,y: output_transform(x,y, scale))
    elif args.command == 'load':
        model, domain_vars, config, scale = load_function(process_type, subtype, output_transform=None if not output_transform else lambda x,y: output_transform(x,y, scale))
    elif args.command == 'list':
        return
    else:
        raise ValueError('UNVALIDEr cOMmAND DU KEK')
        

    ### Train
    callbacks = get_callbacks(config, scale, points_data=point_data, gnd_truth=fem_value_points)

    if hasattr(args, 'epochs') and args.epochs > 0:
        loss_history, train_state = model.train(iterations=args.epochs, callbacks=callbacks.values())
        #np.save('train_state.npy', train_state)
    else:
        loss_history = dde.model.LossHistory()

    ### VIS
    Vis = visualize(args.vis, process_type, subtype, model, loss_history, config, scale, point_data, fem_value_points, callbacks)

    ### SAVE

    #if rank == 0 and args.save:
    save_function(model, domain_vars, loss_history, config, scale, Vis, process_type, subtype)    



def visualize(vis_type, process_type, subtype, model, loss_history, config, scale, point_data, fem_value_points, callbacks):
    vis = {}
    if vis_type in ['loss', 'all']:
        try:
            loss_figures = MAP[process_type][subtype]['vis']['loss'](loss_history, config.loss_labels)
            vis.update(loss_figures)
        except KeyError as e:
            print(f"Warning: Loss visualization not available for {process_type}/{subtype}: {e}")
        except Exception as e:
            print(f"Error generating loss visualization: {e}")

    if vis_type in ['mse', 'all'] and 'dataCollectorCallback' in callbacks:
        mse = callbacks['dataCollectorCallback'].collected_data['mse_history']
        mse_plot = plot_mse(mse)
        vis.update(mse_plot)
    
    if vis_type in ['field', 'all']:
        #try:
            field_figures = MAP[process_type][subtype]['vis']['field'](model, scale, point_data, fem_value_points)
            vis.update(field_figures)
        #except KeyError as e:
        #    print(f"Warning: Field visualization not available for {process_type}/{subtype}: {e}")
        #except Exception as e:
        #    print(f"Error generating field visualization: {e}")
    
    #print(f"Generated visualizations: {list(vis.keys())}")
    return vis