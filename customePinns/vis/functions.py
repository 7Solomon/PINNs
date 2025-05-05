import sys
from config import BConfig
from vis.time_variant import draw_time_dependant_plot, draw_zt_time_dependant_plot
from utils import Domain
from vis.basic import get_pred_grid, get_steady_pred_grid, get_zt_matrix, get_zt_pred_grid, plot_loss, vis_plate_2d


def visualize_field(model, domain: Domain, conf: BConfig):
    if model is None or domain is None:
        print('Model NOne or domain None', file=sys.stderr)
        return
    try:
        if domain.header.__contains__('x') and domain.header.__contains__('y'):
            x = domain.header['x']
            y = domain.header['y']
            if 't' in domain.header.keys():
                t = domain.header['t']
                data = get_pred_grid(model, conf.device, x, y, t, domain.rescale_predictions, n_grid_points=100)
                draw_time_dependant_plot(data)
                return
            data = get_steady_pred_grid(model, conf.device, x, y, domain.rescale_predictions, n_grid_points=100)
            vis_plate_2d(data)
        else:
            data = get_zt_pred_grid(model, conf.device, domain.header['z'], domain.header['t'], domain.rescale_predictions, n_grid_points=100)
            draw_zt_time_dependant_plot(data)
    except KeyError as e:
        print(f'ERROR: Keys passen nicht {e}', file=sys.stderr)


def visualize_loss(loss_history):
    if loss_history is None:
        print('Loss history ist None', file=sys.stderr)
        return
    try:
        plot_loss(loss_history)
    except Exception as e:
        print(f'ERROR: Plotting failed {e}', file=sys.stderr)

