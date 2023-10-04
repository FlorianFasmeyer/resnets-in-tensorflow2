import os
import tensorflow as tf


def name_custom_activation(activation):
    """ Currently, the Tensorflow library does not provide auto incrementation for custom layer names.
        Tensorflow Graphs requires each layer to have a unique name. Manually incrementing layer names
        is annoying. The following trick is the best method to change a layer name. It is especially 
        useful when using tf.keras.utils.plot_model to look at synergistic activation models.
    """
    class_name = tf.keras.layers.Activation.__name__
    tf.keras.layers.Activation.__name__ = f'Activation_{activation.__name__}'
    activation_layer = tf.keras.layers.Activation(activation)
    tf.keras.layers.Activation.__name__ = class_name
    return activation_layer


def regularized_padded_conv(*args, **kwargs):
    return tf.keras.layers.Conv2D(
            *args, 
            **kwargs, padding='same', 
            kernel_regularizer=_regularizer,
            kernel_initializer='he_normal', 
            use_bias=False)


def bn_activate(x, activation=tf.keras.activations.relu):
    x = tf.keras.layers.BatchNormalization()(x)      
    return name_custom_activation(activation)(x)


def shortcut(x, filters, stride, mode):
    if x.shape[-1] == filters:
        return x
    elif mode == 'B':
        return regularized_padded_conv(filters, 1, strides=stride)(x)
    elif mode == 'B_original':
        x = regularized_padded_conv(filters, 1, strides=stride)(x)
        return tf.keras.layers.BatchNormalization()(x)
    elif mode == 'A':
        return tf.pad(tf.keras.layers.MaxPool2D(1, stride)(x) if stride>1 else x,
                      paddings=[(0, 0), (0, 0), (0, 0), (0, filters - x.shape[-1])])
    else:
        raise KeyError("Parameter shortcut_type not recognized!")
    

def original_block(x, filters, stride=1, activation=tf.keras.activations.relu, **kwargs):
    c1 = regularized_padded_conv(filters, 3, strides=stride)(x)
    c2 = regularized_padded_conv(filters, 3)(bn_activate(c1, activation))
    c2 = tf.keras.layers.BatchNormalization()(c2)
    
    mode = 'B_original' if _shortcut_type == 'B' else _shortcut_type
    x = shortcut(x, filters, stride, mode=mode)
    return tf.keras.layers.Activation(activation)(x + c2)
    
    
def preactivation_block(
        x,
        filters,
        stride=1,
        preact_block=False,
        activation=tf.keras.activations.relu, 
        pre_activation=tf.keras.activations.relu, 
        **kwargs):
    
    if preact_block:
        x = flow = bn_activate(x, pre_activation)
    else:
        flow = bn_activate(x, activation)
        
    c1 = regularized_padded_conv(filters, 3, strides=stride)(flow)
    if _dropout:
        c1 = tf.keras.layers.Dropout(_dropout)(c1)
        
    c2 = regularized_padded_conv(filters, 3)(bn_activate(c1, activation))
    x = shortcut(x, filters, stride, mode=_shortcut_type)
    return x + c2


def duplicated_preactivation_block(
        x,
        filters,
        stride=1,
        preact_block=False,
        activation=None,
        activation2=None,
        pre_activation=tf.keras.activations.relu):

    if activation2 is None:
        raise ValueError("The duplicated activation paths preactivation block was made to work with synergistic activation functions! If you do not wish to use two different activation functions, use the preactivation block instead. Example: ´cifar_resnet32(block_type='preactivated'´")
    
    if preact_block:
        x = flow1 = flow2 = bn_activate(x, pre_activation)
    else:
        flow = tf.keras.layers.BatchNormalization()(x)      
        flow1 = name_custom_activation(activation)(flow)
        flow2 = name_custom_activation(activation2)(flow)
        
    # Flow 1.
    flow1 = regularized_padded_conv(filters, 3, strides=stride)(flow1)
    if _dropout:
        flow1 = tf.keras.layers.Dropout(_dropout)(flow1)
        
    flow1 = regularized_padded_conv(filters, 3)(bn_activate(flow1, activation))

    # Flow 2.
    flow2 = regularized_padded_conv(filters, 3, strides=stride)(flow2)
    if _dropout:
        flow2 = tf.keras.layers.Dropout(_dropout)(flow2)
        
    flow2 = regularized_padded_conv(filters, 3)(bn_activate(flow2, activation2))

    
    x = shortcut(x, filters, stride, mode=_shortcut_type)
    return x + (flow1 + flow2)


def bootleneck_block(
        x, 
        filters, 
        stride=1, 
        preact_block=False, 
        activation=tf.keras.activations.relu,
        pre_activation=tf.keras.activations.relu,
        **kwargs):
    
    if preact_block:
        x = flow = bn_activate(x, pre_activation)
    else:
        flow = bn_activate(x, activation)
         
    c1 = regularized_padded_conv(filters//_bootleneck_width, 1)(flow)
    c2 = regularized_padded_conv(filters//_bootleneck_width, 3, strides=stride)(bn_activate(c1, activation))
    c3 = regularized_padded_conv(filters, 1)(bn_activate(c2, activation))
    x = shortcut(x, filters, stride, mode=_shortcut_type)
    return x + c3


def _get_activations(block_type, a1, a2):
    return (a1.__name__, a2.__name__) if 'dup' in block_type.__name__ else a1.__name__


def group_of_blocks(
        x, 
        block_type, 
        num_blocks, 
        filters, 
        stride, 
        block_idx=0,
        activation=tf.keras.activations.relu, 
        activation2=tf.keras.activations.relu,
        pre_activation=tf.keras.activations.relu,
        verbose=0):
    
    global _preact_shortcuts
    preact_block = True if _preact_shortcuts or block_idx == 0 else False

    activation_functions = [_get_activations(block_type, activation, activation2)]
    x = block_type(x, filters, stride, preact_block=preact_block, activation=activation, activation2=activation2, pre_activation=pre_activation)
    for i in range(num_blocks-1): 
        a1, a2 = (activation, activation2) if i % 2 else (activation2, activation)
        activation_functions.append(_get_activations(block_type, a1, a2))
        x = block_type(x, filters, activation=a1, activation2=a2)

    if verbose == 1:
        print(f"Group of {num_blocks} blocks set with activation functions {activation_functions}") 
        
    return x


def Resnet(
        input_shape, 
        n_classes, 
        l2_reg=1e-4, 
        group_sizes=(2, 2, 2), 
        features=(16, 32, 64), 
        strides=(1, 2, 2),
        shortcut_type='B', 
        block_type='preactivated', 
        first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
        dropout=0, 
        cardinality=1, 
        bootleneck_width=4, 
        preact_shortcuts=True,
        activation=tf.keras.activations.relu, 
        activation2=None, 
        pre_activation=None, 
        last_activation=None,
        verbose=0):
    
    global _regularizer, _shortcut_type, _preact_projection, _dropout, _cardinality, _bootleneck_width, _preact_shortcuts
    _bootleneck_width = bootleneck_width # used in ResNeXts and bootleneck blocks
    _regularizer = tf.keras.regularizers.l2(l2_reg)
    _shortcut_type = shortcut_type # used in blocks
    _cardinality = cardinality # used in ResNeXts
    _dropout = dropout # used in Wide ResNets
    _preact_shortcuts = preact_shortcuts

    # When synergistic activation functions aren't used, activation2 defaults to activation.
    if activation2 == None:
        activation2 = activation

    if pre_activation == None:
        pre_activation = activation
        
    if last_activation == None:
        last_activation = activation
    
    block_types = {'preactivated': preactivation_block,
                   'dup_preactivated': duplicated_preactivation_block,
                   'bootleneck': bootleneck_block,
                   'original': original_block}
    
    selected_block = block_types[block_type]
    inputs = tf.keras.layers.Input(shape=input_shape)
    flow = regularized_padded_conv(**first_conv)(inputs)
    
    if block_type == 'original':
        flow = bn_activate(flow, pre_activation)

    previous_activation = None
    for block_idx, (group_size, feature, stride) in enumerate(zip(group_sizes, features, strides)):
        # Prevents two blocks from using the same activation in a row.
        # This happens when one group ends with a1 and the next begins with a1.
        a1, a2 = (activation, activation2) if previous_activation == activation else (activation2, activation)
        previous_activation = a1 if group_size % 2 == 0 else a2
        
        flow = group_of_blocks(
                flow,
                block_type=selected_block,
                num_blocks=group_size,
                block_idx=block_idx,
                filters=feature,
                stride=stride,
                activation=a1, 
                activation2=a2,
                pre_activation=pre_activation,
                verbose=verbose)
    
    if block_type != 'original':
        flow = bn_activate(flow, last_activation)
    
    flow = tf.keras.layers.GlobalAveragePooling2D()(flow)
    outputs = tf.keras.layers.Dense(n_classes, kernel_regularizer=_regularizer)(flow)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def load_weights_func(model, model_name):
    try: model.load_weights(os.path.join('saved_models', model_name + '.tf'))
    except tf.errors.NotFoundError: print("No weights found for this model!")
    return model


def cifar_resnet20(block_type='original', shortcut_type='A', l2_reg=1e-4, load_weights=False, **kwargs):    
    model = Resnet(input_shape=(32, 32, 3), n_classes=10, l2_reg=l2_reg, group_sizes=(3, 3, 3), features=(16, 32, 64),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, 
                   shortcut_type=shortcut_type, block_type=block_type, preact_shortcuts=False, **kwargs)
    
    if load_weights: model = load_weights_func(model, 'cifar_resnet20')
    return model


def cifar_resnet32(block_type='original', shortcut_type='A', l2_reg=1e-4, load_weights=False, **kwargs):    
    model = Resnet(input_shape=(32, 32, 3), n_classes=10, l2_reg=l2_reg, group_sizes=(5, 5, 5), features=(16, 32, 64),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_type=shortcut_type, 
                   block_type=block_type, preact_shortcuts=False)
    if load_weights: model = load_weights_func(model, 'cifar_resnet32')
    return model


def cifar_resnet44(block_type='original', shortcut_type='A', l2_reg=1e-4, load_weights=False, **kwargs):    
    model = Resnet(input_shape=(32, 32, 3), n_classes=10, l2_reg=l2_reg, group_sizes=(7, 7, 7), features=(16, 32, 64),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_type=shortcut_type, 
                   block_type=block_type, preact_shortcuts=False)
    if load_weights: model = load_weights_func(model, 'cifar_resnet44')
    return model


def cifar_resnet56(block_type='original', shortcut_type='A', l2_reg=1e-4, load_weights=False, **kwargs):    
    model = Resnet(input_shape=(32, 32, 3), n_classes=10, l2_reg=l2_reg, group_sizes=(9, 9, 9), features=(16, 32, 64),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_type=shortcut_type, 
                   block_type=block_type, preact_shortcuts=False)
    if load_weights: model = load_weights_func(model, 'cifar_resnet56')
    return model


def cifar_resnet110(block_type='preactivated', shortcut_type='B', l2_reg=1e-4, load_weights=False, **kwargs):    
    model = Resnet(input_shape=(32, 32, 3), n_classes=10, l2_reg=l2_reg, group_sizes=(18, 18, 18), features=(16, 32, 64),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_type=shortcut_type, 
                   block_type=block_type, preact_shortcuts=False)
    if load_weights: model = load_weights_func(model, 'cifar_resnet110')
    return model


def cifar_resnet164(shortcut_type='B', load_weights=False, l2_reg=1e-4, **kwargs):    
    model = Resnet(input_shape=(32, 32, 3), n_classes=10, l2_reg=l2_reg, group_sizes=(18, 18, 18), features=(64, 128, 256),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_type=shortcut_type, 
                   block_type='bootleneck', preact_shortcuts=True)
    if load_weights: model = load_weights_func(model, 'cifar_resnet164')
    return model


def cifar_resnet1001(shortcut_type='B', load_weights=False, l2_reg=1e-4, **kwargs):    
    model = Resnet(input_shape=(32, 32, 3), n_classes=10, l2_reg=l2_reg, group_sizes=(111, 111, 111), features=(64, 128, 256),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_type=shortcut_type, 
                   block_type='bootleneck', preact_shortcuts=True)
    if load_weights: model = load_weights_func(model, 'cifar_resnet1001')
    return model


def cifar_wide_resnet(N, K, block_type='preactivated', shortcut_type='B', dropout=0, l2_reg=2.5e-4, **kwargs):    
    assert (N-4) % 6 == 0, "N-4 has to be divisible by 6"
    lpb = (N-4) // 6 # layers per block - since N is total number of convolutional layers in Wide ResNet
    model = Resnet(input_shape=(32, 32, 3), n_classes=10, l2_reg=l2_reg, group_sizes=(lpb, lpb, lpb), features=(16*K, 32*K, 64*K),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_type=shortcut_type,
                   block_type=block_type, dropout=dropout, preact_shortcuts=True)
    return model


def cifar_WRN_16_4(shortcut_type='B', load_weights=False, dropout=0, l2_reg=2.5e-4, **kwargs):    
    model = cifar_wide_resnet(16, 4, 'preactivated', shortcut_type, dropout=dropout, l2_reg=l2_reg)
    if load_weights: model = load_weights_func(model, 'cifar_WRN_16_4')
    return model


def cifar_WRN_40_4(shortcut_type='B', load_weights=False, dropout=0, l2_reg=2.5e-4, **kwargs):    
    model = cifar_wide_resnet(40, 4, 'preactivated', shortcut_type, dropout=dropout, l2_reg=l2_reg)
    if load_weights: model = load_weights_func(model, 'cifar_WRN_40_4')
    return model


def cifar_WRN_16_8(shortcut_type='B', load_weights=False, dropout=0, l2_reg=2.5e-4, **kwargs):    
    model = cifar_wide_resnet(16, 8, 'preactivated', shortcut_type, dropout=dropout, l2_reg=l2_reg)
    if load_weights: model = load_weights_func(model, 'cifar_WRN_16_8')
    return model


def cifar_WRN_28_10(shortcut_type='B', load_weights=False, dropout=0, l2_reg=2.5e-4, **kwargs):    
    model = cifar_wide_resnet(28, 10, 'preactivated', shortcut_type, dropout=dropout, l2_reg=l2_reg)
    if load_weights: model = load_weights_func(model, 'cifar_WRN_28_10')
    return model


def cifar_resnext(N, cardinality, width, shortcut_type='B', **kwargs):    
    assert (N-3) % 9 == 0, "N-4 has to be divisible by 6"
    lpb = (N-3) // 9 # layers per block - since N is total number of convolutional layers in Wide ResNet
    model = Resnet(input_shape=(32, 32, 3), n_classes=10, l2_reg=1e-4, group_sizes=(lpb, lpb, lpb), features=(16*width, 32*width, 64*width),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1}, shortcut_type=shortcut_type,
                   block_type='resnext', cardinality=cardinality, width=width)
    return model