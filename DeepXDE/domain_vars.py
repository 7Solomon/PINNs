from utils.metadata import Domain


transient_heat_2d_domain = Domain(spatial={
        'x': (0, 0.5),
        'y': (0, 1)
    }, temporal={
        't': (0, 8.6e4)
    },
    resolution={'x': 100, 'y': 50, 't': 20}
    )
steady_heat_2d_domain = Domain(spatial={
        'x': (0, 2),
        'y': (0, 1)
    }, temporal=None,
    resolution={'x': 100, 'y': 50}
    )

einspannung_2d_domain = Domain(
        spatial={
            #'x':(0,25),
            'x':(0,10),
            'y':(0,1)
        },
       resolution={'x': 250, 'y': 25}
    )
fest_lost_domain = Domain(
        spatial={
            'x':(0,1)
        },
        resolution={'x': 100}
    )

moisture_1d_domain = Domain(
        spatial={
            'z': (0, 1),
        }, temporal={
            #'t': (0, 1.3e7)  #4e6 seconds started  makes time derivative WAAAAAAY to small which leads to no learning there just steady output whihc is pretty interesting
            't': (0, 18.12e5) # ca 3 week
        #    't': (0, 8.6e4)  # ca 1 day
            #'t': (0, 7.2e3)  # ca 2 hour
            #'t': (0, 1.4e3)  # ca 20 minutes
            #'t': (0, 3e2)  # ca 5 minutes
        },
        #resolution={'z': 20, 't': 20}
        resolution={'z': 50, 't': 20}
    )
moisture_2d_domain = Domain(
        spatial={
            'x': (0, 1),
            'z': (0, 1),
        }, temporal=None,
        resolution={'x': 100, 'y': 100}
    )

thermal_mechanical_2d_domain = Domain(
        spatial={
            'x': (0, 0.1),
            'y': (0, 1)
        }, temporal={
           't': (0,1250) 
           # 't': (0, 3.6e3)
        },
        resolution={'x': 10, 'y': 100, 't': 50}
    )
thermal_moisture_2d_domain = Domain(
        spatial={
            'x': (0, 0.25),
            'y': (0, 1)
        }, temporal={
            't': (0, 2e6)  # ca 3 days i think
            #'t': (0, 3.6e3)  # ca 1 hour
        },
        resolution={'x': 100, 'y': 100, 't': 20}
    )
mechanical_moisture_2d_domain = Domain(
        spatial={
            'x': (0, 10),
            'y': (0, 1)
        }, temporal={
        #    't': (0, 4e6)  # ca 1 week
            't': (0, 8.6e4)  # ca 1 day
        },
        resolution={'x': 100, 'y': 50, 't': 20}
    )
