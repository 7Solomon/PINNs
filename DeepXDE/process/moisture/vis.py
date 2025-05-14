from vis import visualize_time_dependent_field


def visualize_field(model, type, inverse_scale=None):
    if type == '1d_head':
        domain_stuff = {
            'x_min': 0,
            'x_max': 1,
            'y_min': 0,
            'y_max': 2,
            't_min': 0,
            't_max': 1,
            'min_val': 0.0,
            'max_val': 1.0
        }
        visualize_time_dependent_field(model, domain_stuff,inverse_scale=inverse_scale, animate=True)