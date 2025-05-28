from utils.metadata import Domain


transient_heat_2d_domain = Domain(spatial={
        'x': (0, 0.5),
        'y': (0, 1)
    }, temporal={
        't': (0, 4e6)
    })
steady_heat_2d_domain = Domain(spatial={
        'x': (0, 2),
        'y': (0, 1)
    }, temporal=None)

fest_lost_2d_domain = Domain(
        spatial={
            'x':(0,10),
            'y':(0,1)
        }
    )
einspannung_2d_domain = Domain(
        spatial={
            'x':(0,10),
            'y':(0,1)
        }
    )
fest_lost_domain = Domain(
        spatial={
            'x':(0,1)
        }
    )

moisture_1d_mixed_domain = Domain(
        spatial={
            'z': (0, 1),
        }, temporal={
            't': (0, 1e10)
        }
    )