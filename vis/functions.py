import sys
from vis.basic import get_pred_grid, get_steady_pred_grid, plot_loss, vis_plate_2d
from vars import device

def visualize_field(model, domain):
    if model is None or domain is None:
        print('Model NOne or domain None', file=sys.stderr)
        return
    try:
        x = next(iter(domain.header['x']))
        y = next(iter(domain.header['y']))
        if 't' in domain.header.keys():
            t = next(iter(domain.header['t']))
            data = get_pred_grid(model, device, x, y, t, n_grid_points=100)
            vis_plate_2d(data)
            return
        raise NotImplementedError('HIER FEHLT CALLBACK FUNCTION, ka wie')
        data = get_steady_pred_grid(model, device, x, y, n_grid_points=100)
        vis_plate_2d(data)
    except KeyError as e:
        print('ERROR: Keys passen nicht', file=sys.stderr)
    except Exception as e:
        print(f'Eine andere Exception ist aufgetreten: {e}', file=sys.stderr)


def visualize_loss(loss_history):
    if loss_history is None:
        print('Loss history ist None', file=sys.stderr)
        return
    try:
        plot_loss(loss_history)
    except Exception as e:
        print(f'ERROR: Plotting failed', file=sys.stderr)
