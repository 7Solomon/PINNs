import domain_vars
import process
import config
import process.heat.output_transform
import process.thermal_mechanical
import vis

MAP = {
    'mechanic':{
        'fest_los': {
            'domain': process.mechanic.domain.get_fest_los_domain,
            'domain_vars': domain_vars.fest_lost_domain,
            'config': config.bernoulliBalkenConfig,
            'scale': process.mechanic.scale.MechanicScale,
            'vis': {
                'loss': vis.plot_loss,
                'field': process.mechanic.vis.visualize_field_1d,
                #'div': process.mechanic.vis.visualize_divergence,
            },
            'path': 'models/mechanic/fest_los',
        },
        'einspannung_2d':{
            'domain': process.mechanic.domain.get_einspannung_domain_2d,
            'domain_vars': domain_vars.einspannung_2d_domain,
            'config': config.bernoulliBalken2DConfig,
            'scale' : process.mechanic.scale.MechanicScale,
            'gnd_function':  process.mechanic.gnd.get_einspannung_2d_fem_points,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.mechanic.vis.visualize_field_2d,
                #'div': process.mechanic.vis.visualize_divergence,
            },
            'path': 'models/mechanic/einspannung_2d',
        
        },
        'einspannung_2d_ensemble':{
            'domain': process.mechanic.domain.get_einspannung_domain_2d_ensamble,
            'domain_vars': domain_vars.einspannung_2d_domain,
            'config': config.bernoulliBalken2DEnsembleConfig,
            'gnd_function':  process.mechanic.gnd.get_ensemble_einspannung_2d_fem_points,

            'scale' : process.mechanic.scale.EnsemnbleMechanicScale,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.mechanic.vis.visualize_field_2d_ensemble,
                #'div': process.mechanic.vis.visualize_divergence,
            },
            'path': 'models/mechanic/einspannung_2d_ensemble',
        
        },
        #'fest_los_2d': {
        #    'domain': process.mechanic.domain.get_fest_los_domain_2d,
        #    'config': config.BernoulliBalken2DConfig,
        #    'vis' : {
        #        'loss': vis.plot_loss,
        #        'field': process.mechanic.vis.visualize_field_2d,
        #        #'div': process.mechanic.vis.visualize_divergence,
        #    },
        #    'path': 'models/mechanic/fest_los_2d',
        #},
       'einspannung': {
            'domain': process.mechanic.domain.get_einspannung_domain,
            'domain_vars': domain_vars.einspannung_2d_domain,
            'config': config.bernoulliBalkenConfig,
            'scale' : process.mechanic.scale.MechanicScale,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.mechanic.vis.visualize_field_1d,
                #'div': process.mechanic.vis.visualize_divergence,
            },
            'path': 'models/mechanic/einspannung',
        },
        'fest_los_t': {
            'domain': process.mechanic.domain.get_fest_los_t_domain,
            #'domain_vars': domain_vars.fest_lost_domain,
            'config': config.bernoulliBalkenTConfig,
            'scale': process.mechanic.scale.MechanicScale,

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
            'scale' : process.mechanic.scale.MechanicScale,
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
            'domain_vars': domain_vars.steady_heat_2d_domain,
            'config': config.steadyHeatConfig,
            'scale' : process.heat.scale.Scale,
            #'output_transform': process.heat.output_transform.output_transform,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.heat.vis.visualize_steady_field,
                #'div': process.heat.vis.visualize_divergence,
            },
            'path': 'models/heat/steady',
        },
        'transient': {
            'domain': process.heat.domain.get_transient_domain,
            'domain_vars': domain_vars.transient_heat_2d_domain,
            'config': config.transientHeatConfig,
            'scale' : process.heat.scale.Scale,
            'gnd_function': process.heat.gnd.get_transient_fem_points,
            #'gnd_function': process.heat.gnd.get_transient_analytical_solution,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.heat.vis.visualize_transient_field,
                #'div': process.heat.vis.visualize_divergence,
            },
            'path': 'models/heat/transient',
        }
    },
    'moisture':{
        '1d_mixed': {
            'domain': process.moisture.domain.get_1d_mixed_domain,
            'domain_vars': domain_vars.moisture_1d_domain,
            'config': config.richardsMixed1DConfig,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.moisture.vis.visualize_2d_mixed,

                #'div': process.moisture.vis.visualize_divergence,
            },
            'path': 'models/moisture/1d_mixed',
        },
        '1d_head': {
            'domain': process.moisture.domain.get_1d_head_domain,
            'domain_vars': domain_vars.moisture_1d_domain,
            'config': config.richards1DConfig,
            'scale' : process.moisture.scale.HeadScale,
            'output_transform': process.moisture.output_transform.output_transform_1d_head,
            'gnd_function': process.moisture.gnd.get_richards_1d_head_fem_points,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.moisture.vis.vis_1d_head,
                #'div': process.moisture.vis.visualize_divergence,
            },
            'path': 'models/moisture/1d_head',
        },
        '1d_saturation': {
            'domain': process.moisture.domain.get_1d_saturation_domain,
            'domain_vars': domain_vars.moisture_1d_domain,
            'config': config.richards1DConfig,
            'output_transform': process.moisture.output_transform.output_transform_1d_saturation,
            'scale' : process.moisture.scale.SaturationScale,
            'gnd_function': process.moisture.gnd.get_richards_1d_saturation_fem_points,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.moisture.vis.vis_1d_saturation,
                'kwargs': {
                    'title': '1D Saturation Field',
                }
                #'div': process.moisture.vis.visualize_divergence,
            },
            'path': 'models/moisture/1d_saturation',
        },
        '2d_darcy': {
            'domain': process.moisture.domain.get_2d_darcy_domain,
            'domain_vars': domain_vars.moisture_2d_domain,
            'config': config.darcy2DConfig,
            #'scale' : process.moisture.scale.Scale,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.moisture.vis.visualize_2d_darcy,
                #'div': process.moisture.vis.visualize_divergence,
            },
            'path': 'models/moisture/2d_darcy',
        }
    },
    'thermal_mechanical': {
        '2d': {
            'domain': process.thermal_mechanical.domain.get_thermal_2d_domain,
            'domain_vars': domain_vars.thermal_mechanical_2d_domain,
            'config': config.thermalMechanical2DConfig,
            'scale' : process.thermal_mechanical.scale.Scale,
            'gnd_function': process.thermal_mechanical.gnd.get_thermal_mechanical_fem_points,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.thermal_mechanical.vis.vis_2d_multi,
                'kwargs': {
                    #'variable_indices' : [0,1,2,3],
                    #'show_displacement_comparison': True,
                    #'displacement_amplifier': 1000,
                }
                #'div': process.thermal_mechanical.vis.visualize_divergence,
            },
            'path': 'models/thermal_mechanical/2d',
        }
    },
    'thermal_moisture': {
        '2d': {
            'domain': process.thermal_moisture.domain.get_2d_domain,
            'domain_vars': domain_vars.thermal_moisture_2d_domain,
            'config': config.thermalMoisture2DConfig,
            'scale' : process.thermal_moisture.scale.Scale,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.thermal_moisture.vis.vis_2d_multi,
                #'field': lambda x :print('No field visualization implemented for thermal moisture 2D yet, you '),
                #'div': process.thermal_moisture.vis.visualize_divergence,
            },
            'path': 'models/thermal_moisture/2d',
        }
    },
    'mechanical_moisture': {
        '2d': {
            'domain': process.mechanical_moisture.domain.domain_2d,
            'domain_vars': domain_vars.mechanical_moisture_2d_domain,
            'config': config.mechanicalMoisture2DConfig,
            'scale' : process.mechanical_moisture.scale.Scale,
            'output_transform': process.mechanical_moisture.output_transform.output_transform_2d,
            'vis' : {
                'loss': vis.plot_loss,
                'field': process.mechanical_moisture.vis.visualize_transient_mechanical_moisture_comparison,
                #'div': process.mechanical_moisture.vis.visualize_divergence,
            },
            'path': 'models/mechanical_moisture/2d',
        }
    }

}