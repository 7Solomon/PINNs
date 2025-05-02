
import argparse
import sys


def manage_args(args: argparse.Namespace):
    if args.command == 'add':
        if args.type == 'steady_heat':
            from heat.manager import create_steady
            create = create_steady
        elif args.type == 'transient_heat':
            from heat.manager import create_transient
            create = create_transient
        else:
            print('Unbekannter Typ', file=sys.stderr)
            return
        model, cData = create()

    elif args.command == 'load':
        if args.type == 'steady_heat':
            from heat.manager import load
            load = load
        elif args.type == 'transient_heat':
            from heat.manager import load
            print('HIER NICHT IMPLEMENTIERT, wahrscheinlich')
            load = load
        else:
            print('Unbekannter Typ', file=sys.stderr)
            return
        model, cData = load()
    
    # Vis
    if model and args.vis:
        if args.vis == 'loss':
            from vis.functions import visualize_loss
            visualize_loss(cData.loss)
        elif args.vis == 'field':
            from vis.functions import visualize_field
            visualize_field(model, cData.domain)
        elif args.vis == 'all':
            from vis.functions import visualize_loss, visualize_field, visualize_model
            visualize_loss(cData.loss)
            visualize_field(model, cData.domain)
    elif model is None:
        print('beim Laden ist ein Fehler passiert', file=sys.stderr)

