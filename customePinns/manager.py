
import argparse
import sys



def manage_args(args: argparse.Namespace):
    if args.command == 'add':
        if args.type == 'steady_heat':
            #from heat.manager import create_steady
            #create = create_steady
            from config import SteadyHeatConfig
            from process.heat.domain import generate_steady_domain
            from nn_stuff.model import create_model
            conf = SteadyHeatConfig()
            domain = generate_steady_domain(conf)
            model, cData = create_model(domain, conf)
        elif args.type == 'transient_heat':
            raise NotImplementedError('Noch nicht implementiert, du kek')
        elif args.type == 'moisture':
            from config import MoistureConfig
            from process.moisture.normal_pinn_2d.domain import get_domain
            from nn_stuff.model import create_model
            conf = MoistureConfig()
            domain = get_domain()
            model, cData = create_model(domain, conf)
        elif args.type == 'moisture_HB':
            from config import MoistureHeadBodyConfig
            from process.moisture.head_body_pinn_1d.domain import get_domain
            from nn_stuff.model import create_model
            conf = MoistureHeadBodyConfig()
            domain = get_domain(conf)
            model, cData = create_model(domain, conf)
        elif args.type == 'mechanic':
            from config import BernoulliBalkenConfig
            from process.mechanic.domain import get_domain
            from nn_stuff.model import create_model
            conf = BernoulliBalkenConfig()
            domain = get_domain(conf)
            model, cData = create_model(domain, conf)
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
        elif args.type == 'moisture_HB':
            from config import MoistureHeadBodyConfig
            conf = MoistureHeadBodyConfig()
            model, cData = load_model(conf)
        elif args.type == 'mechanic':
            from config import BernoulliBalkenConfig
            conf = BernoulliBalkenConfig()
            model, cData = load_model(conf)
        else:
            print('Unbekannter Typ', file=sys.stderr)
            return
    elif args.command == 'test':
        if args.type == 'functions':
            from vis.basic import plot_saturation
            plot_saturation(conf)
        elif args.type == 'COMSOL':
            from nn_stuff.model import load_model
            from config import SteadyHeatConfig
            from vis.div_ground_truth import load_COMSOL_file_data, analyze_COMSOL_file_data, vis_diffrence
            conf = SteadyHeatConfig()
            model, cData = load_model(conf)
            data = load_COMSOL_file_data('heat\COMSOL\comsolOutFile.txt')
            dom, val= analyze_COMSOL_file_data(data)
            vis_diffrence(model, cData.domain, dom, val)
        elif args.type == 'domain':
            from domain.heat import get_steady_heat_domain
            from nn_stuff.model import create_model
            from config import SteadyHeatConfig
            conf = SteadyHeatConfig()
            domain = get_steady_heat_domain(conf)
            #model, cData = create_model(domain, conf)
        elif args.type == 'field':
            from process.mechanic.domain import get_domain
            from vis.basic import visualize_beam_deflection
            from config import BernoulliBalkenConfig
            from nn_stuff.model import load_model
            from vis.analytic import bernoulli_beam_deflection

            conf = BernoulliBalkenConfig()

            model, cData = load_model(conf)
            visualize_beam_deflection(model,  cData.domain, analytical_func=bernoulli_beam_deflection)
        else:
            print('Unbekannter Typ', file=sys.stderr)
            return
    
    # Vis
    if args.vis and model:
        if args.vis == 'loss':
            from vis.functions import visualize_loss
            visualize_loss(cData.loss)
        elif args.vis == 'field':
            from vis.functions import visualize_field
            visualize_field(model, cData.domain, conf)
        elif args.vis == 'all':
            from vis.functions import visualize_loss, visualize_field
            visualize_loss(cData.loss)
            visualize_field(model, cData.domain, conf)
        

