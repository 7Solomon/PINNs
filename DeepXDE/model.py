from utils.MBNN import MBNN
from utils.fourier_features import FourierFeatureTransform

from utils.metadata import BConfig
import deepxde as dde
import numpy as np


def create_flexible_model(data, config: BConfig, output_transform=None):
    model_type = config.get('model_type', 'FNN')
    
    if model_type == 'FNN':
        net = create_fnn_model(config)
    elif model_type == 'DeepONet':
        net = create_deeponet_model(config)
    elif model_type == 'MultiBranch':
        net = create_multibranch_model(config)
    elif model_type == 'Custom':
        net = create_custom_model(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Apply transforms
    if config.get('fourier_transform_features'):
        feature_transform = FourierFeatureTransform(
            in_dim=config.input_dim, 
            num_features=config.fourier_transform_features, 
            sigma=config.get('fourier_sigma', 1.0)
        )
        net.apply_feature_transform(feature_transform)
    
    if output_transform:
        net.apply_output_transform(output_transform)
    
    model = dde.Model(data, net)
    
    return compile_model(model, config)

def create_fnn_model(config):
    """Create standard Feed-Forward Neural Network"""
    hidden = config.get('layers', [50]*4)
    fourier = config.get('fourier_transform_features', 1)
    layers = [config.input_dim*fourier] + hidden + [config.output_dim]
    activation = config.get('activation', 'tanh')
    initializer = config.get('initializer', 'Glorot uniform')
    
    return dde.maps.FNN(layers, activation, initializer)

def create_deeponet_model(config):
    raise NotImplementedError("DeepONet model creation is not implemented yet.")
#    """Create DeepONet architecture"""
#    hidden_branch_layers = config.get('branch_layers', [100] + [50]*3 + [50])
#    hidden_trunk_layers = config.get('trunk_layers', [config.input_dim] + [50]*3 + [50])
#
#    activation = config.get('activation', 'tanh')
#    initializer = config.get('initializer', 'Glorot uniform')
#    
#    return dde.maps.DeepONet(
#        branch_layers, trunk_layers, activation, initializer,
#        use_bias=config.get('use_bias', True),
#        stacked=config.get('stacked', True)
#    )

def create_multibranch_model(config):
    """Create multi-branch network for multiphysics problems"""
    # Define branch configurations
    branches = config.get('branches', {})
    branch_definitions = []
    for branch_name, branch in branches.items():
        input_indices = branch.get('input_indices', [])
        layer_dims = branch.get('layer_dims', [])
        branch_definitions.append((input_indices, layer_dims))

    return MBNN(branch_definitions)

def create_custom_model(config):
    """Create custom architecture based on config"""
    raise NotImplementedError("Custom model creation is not implemented yet.")
    #custom_config = config.get('custom_architecture', {})
    #
    ## Example: Residual connections, attention mechanisms, etc.
    #if custom_config.get('type') == 'ResNet':
    #    return create_resnet_model(custom_config)
    #elif custom_config.get('type') == 'Attention':
    #    return create_attention_model(custom_config)
    #else:
    #    # Fallback to standard FNN
    #    return create_fnn_model(config)

def compile_model(model, config):
    """Enhanced model compilation with multiple optimization strategies"""
    optimizer = config.compile_args.get("optimizer")

    #compile_strategy = config.get('compile_strategy', 'standard')
    #if compile_strategy == 'lbfgs':
    #    return compile_with_lbfgs(model, config)
    #elif compile_strategy == 'adaptive':
    #    return compile_with_adaptive_training(model, config)
    #else:
    model.compile(**config.compile_args, loss_weights=config.loss_weights, 
                decay=config.get('decay'))
    return model

#def compile_with_lbfgs(model, config):
#    """Compile for Adam -> LBFGS training"""
#    model.compile(optimizer='adam', lr=config.get('adam_lr', 0.001), 
#                 loss_weights=config.loss_weights)
#    
#    model._lbfgs_config = {
#        'iterations': config.get('lbfgs_iterations', 5000),
#        'switch_epoch': config.get('lbfgs_switch_epoch', 10000)
#    }
#    
#    return model



#### Callbacks for training management
def get_callbacks(config, scale, points_data=None, gnd_truth=None):
    callbacks = {}
    if hasattr(config, 'callbacks') and 'slowPdeAnnealing' in config.callbacks:
        from utils.dynamic_loss import SlowPdeLossWeightCallback
        callbacks.update({'slowPdeLossWeightCallback': SlowPdeLossWeightCallback(pde_indices=config.pde_indices, final_weight=config.annealing_value)})
    if hasattr(config, 'callbacks') and 'dynamicLossWeight' in config.callbacks:
        from utils.dynamic_loss import DynamicLossWeightCallback
        callbacks.update({'dynamicLossWeightCallback': DynamicLossWeightCallback()})
    if hasattr(config, 'callbacks') and 'resample' in config.callbacks:
        callbacks.update({'PDEPointResampler': dde.callbacks.PDEPointResampler(period=1000)})
    if hasattr(config, 'callbacks') and 'variable_lr_config' in config.callbacks:
        from utils.variable_lr import VariableLearningRateCallback
        callbacks.update({'variableLearningRateCallback': VariableLearningRateCallback(**config.variable_lr_config)})
    if hasattr(config, 'callbacks') and 'dataCollector' in config.callbacks:
        from utils.callbacks.debug_callback import DataCollectorCallback
        callbacks.update({'dataCollectorCallback': DataCollectorCallback(points_data=points_data, gnd_truth=gnd_truth, scale=scale)})
    return callbacks