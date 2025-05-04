
import argparse
import sys


def manage_args(args: argparse.Namespace):
    if args.command == 'add':
        if args.type == 'steady_heat':
            #from heat.manager import create_steady
            #create = create_steady
            from config import SteadyHeatConfig
            from heat.domain import generate_steady_domain
            from nn_stuff.model import create_model
            conf = SteadyHeatConfig()
            domain = generate_steady_domain(conf)
            model, cData = create_model(domain, conf)
        elif args.type == 'transient_heat':
            from heat.manager import create_transient
            create = create_transient
        elif args.type == 'moisture':
            from moisture.manager import create
            create = create
        else:
            print(f'Unbekannter Typ: {args.type}', file=sys.stderr)
            return


    elif args.command == 'load':
        from nn_stuff.model import load_model
        if args.type == 'steady_heat':
            from config import SteadyHeatConfig
            conf = SteadyHeatConfig()
            model, cData = load_model(conf)
        elif args.type == 'transient_heat':
            raise NotImplementedError('Noch nicht implementiert, du kek')
        elif args.type == 'moisture':
            from config import MoistureConfig
            conf = MoistureConfig()
            model, cData = load_model(conf)
        else:
            print('Unbekannter Typ', file=sys.stderr)
            return
    elif args.command == 'test':
        if args.type == 'functions':
            from moisture.manager import vis_functions
            vis_functions()
        elif args.type == 'COMSOL':
            from nn_stuff.model import load_model
            from config import SteadyHeatConfig
            from vis.div_ground_truth import load_COMSOL_file_data, analyze_COMSOL_file_data, vis_diffrence
            conf = SteadyHeatConfig()
            model, cData = load_model(conf)
            data = load_COMSOL_file_data('heat\COMSOL\comsolOutFile.txt')
            dom, val= analyze_COMSOL_file_data(data)
            vis_diffrence(model, cData.domain, dom, val)


        return
    
    # Vis
    if model and args.vis:
        if args.vis == 'loss':
            from vis.functions import visualize_loss
            visualize_loss(cData.loss)
        elif args.vis == 'field':
            from vis.functions import visualize_field
            visualize_field(model, cData.domain)
        elif args.vis == 'all':
            from vis.functions import visualize_loss, visualize_field
            visualize_loss(cData.loss)
            visualize_field(model, cData.domain)
        

    elif model is None:
        print('beim Laden ist ein Fehler passiert', file=sys.stderr)

