import process
import config
import vis

MAP = {
    'mechanic':{
        'fest_los': {
            'domain': process.mechanic.domain.get_fest_los_domain,
            'config': config.bernoulliBalkenConfig,
            'vis': {
                'loss': vis.plot_loss,
                'field': process.mechanic.vis.visualize_field_1d,
                #'div': process.mechanic.vis.visualize_divergence,
            },
            'path': 'models/mechanic/fest_los',
        },
        'fest_los_2d': {
            'domain': process.mechanic.domain.get_fest_los_domain_2d,
            'config': config.BernoulliBalken2DConfig,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.mechanic.vis.visualize_field_2d,
                #'div': process.mechanic.vis.visualize_divergence,
            },
            'path': 'models/mechanic/fest_los_2d',
        },
       'einspannung': {
            'domain': process.mechanic.domain.get_einspannung_domain,
            'config': config.bernoulliBalkenConfig,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.mechanic.vis.visualize_field_1d,
                #'div': process.mechanic.vis.visualize_divergence,
            },
            'path': 'models/mechanic/einspannung',
        },
        'fest_los_t': {
            'domain': process.mechanic.domain.get_fest_los_t_domain,
            'config': config.bernoulliBalkenTConfig,
            'vis' : {
                'loss': vis.plot_loss,
                #'field': process.mechanic.vis.,
                #'div': process.mechanic.vis.visualize_divergence,
            },
            'path': 'models/mechanic/fest_los_t',
        },
        'cooks': {
            'domain': process.mechanic.domain.get_cooks_domain,
            'config': config.cooksMembranConfig,
            'vis' : {
                'loss': vis.plot_loss,
                #'field': process.mechanic.vis.visualize_field,
                #'div': process.mechanic.vis.visualize_divergence,
            },
            'path': 'models/mechanic/cooks_membrane',
        }
    },
    'heat':{
        'steady': {
            'domain': process.heat.domain.get_steady_domain,
            'config': config.steadyHeatConfig,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.heat.vis.visualize_steady_field,
                #'div': process.heat.vis.visualize_divergence,
            },
            'path': 'models/heat/steady',
        },
        'transient': {
            'domain': process.heat.domain.get_transient_domain,
            'config': config.transientHeatConfig,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.heat.vis.visualize_steady_field,
                #'div': process.heat.vis.visualize_divergence,
            },
            'path': 'models/heat/transient',
        }
    },
    'moisture':{
        '1d_head': {
            'domain': process.moisture.domain.get_1d_domain,
            'config': config.richards1DConfig,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.moisture.vis.vis_1d_head,
                #'div': process.moisture.vis.visualize_divergence,
            },
            'path': 'models/moisture/1d_head',
        },
    }
}