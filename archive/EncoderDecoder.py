import tensorflow as tf
from tensorflow import keras


def DenseEncoder(units_full=128, num_layers=3, 
                 actlay_config=dict(activation="elu",
                                    kernel_initializer='he_normal'),
                 linlay_config=dict(activation=None),
                 add_init_fin=True, **kwargs):
    input_layer = keras.layers.Input(shape=[units_full])
    x = keras.layers.Dense(units_full, name='hidden1', **actlay_config)(input_layer)
    for i in range(num_layers-2):
        x = keras.layers.Dense(units_full, 
                               name='hidden{}'.format(i+2), 
                               **actlay_config)(x)
    output_layer = keras.layers.Dense(units_full, 
                                      name='hidden{}'.format(num_layers), 
                                      **linlay_config)(x)
    if add_init_fin:
        output_layer = keras.layers.add([output_layer, input_layer])
    return keras.Model(input_layer,output_layer, **kwargs)

def DenseDecoder(units_full=128, num_layers=3, 
                 actlay_config=dict(activation="elu",
                                    kernel_initializer='he_normal'),
                 linlay_config=dict(activation=None),
                 add_init_fin=True, **kwargs):
    input_layer = keras.layers.Input(shape=[units_full])
    x = keras.layers.Dense(units_full, name='hidden1', **actlay_config)(input_layer)
    for i in range(num_layers-2):
        x = keras.layers.Dense(units_full, 
                               name='hidden{}'.format(i+2), 
                               **actlay_config)(x)
    output_layer = keras.layers.Dense(units_full, 
                                      name='hidden{}'.format(num_layers), 
                                      **linlay_config)(x)
    if add_init_fin:
        output_layer = keras.layers.add([output_layer, input_layer])
    return keras.Model(input_layer,output_layer, **kwargs)
