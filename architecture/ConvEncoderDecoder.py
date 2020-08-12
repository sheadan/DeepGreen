import tensorflow as tf
import copy

class ConvEncoderDecoder(tf.keras.Model):
    """A Convolutional layer-based encoder."""
    def __init__(self, units_full=128,
                 num_filters = [8, 16, 32, 64], 
                 convlay_config=dict(kernel_size=4, strides=1, padding='SAME',
                                     activation='relu',
                                     kernel_initializer='he_normal'),
                 poollay_config=dict(pool_size=2, strides=2, padding='VALID'),
                 actlay_config=dict(activation='relu',
                                    kernel_initializer='he_normal'),
                 linlay_config=dict(activation=None),
                 add_init_fin=True, **kwargs):
        super().__init__(**kwargs)

        # Construct a list of the convolutional and pooling layers used in this block
        self.conv_layers = [tf.keras.layers.Conv1D(filters=num_filters[0],
                                                   **convlay_config)]
        for filters in num_filters[1:]:
            self.conv_layers.append(tf.keras.layers.AveragePooling1D(**poollay_config))
            self.conv_layers.append(tf.keras.layers.Conv1D(filters=filters, 
                                                           **convlay_config))
        
        # Construct a list of the dense layers in this block
        self.dense_layers = [tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(units_full, 
                                                   **actlay_config),
                             tf.keras.layers.Dense(units_full,
                                                   **linlay_config)] 

        # The boolean for adding the initial and final layers together
        self.add_init_fin = add_init_fin

    def call(self, input_tensor):
        x = tf.expand_dims(input_tensor, axis=-1)
        for layer in self.conv_layers:
            x = layer(x)
        for layer in self.dense_layers:
            x = layer(x)
        if self.add_init_fin:
            x += input_tensor
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "num_filters": self.num_layers,
                "actlay_config": self.actlay_config,
                "linlay_config": self.linlay_config,
                "convlay_config": self.convlay_config,
                "poollay_config": self.poollay_config,
                "units_full": self.units_full,
                "layers": self.layers,
                "add_init_fin": self.add_init_fin}

class ConvDecoder(tf.keras.Model):
    """A Convolutional layer-based encoder."""
    def __init__(self, units_full=128,
                 init_size = 16,
                 num_filters = [64, 32, 16, 8], 
                 deconvlay_config=dict(kernel_size=4, strides=2, padding='SAME',
                                       activation='relu',
                                       kernel_initializer='he_normal'),
                 actlay_config=dict(activation='relu',
                                    kernel_initializer='he_normal'),
                 add_init_fin=True, **kwargs):
        super().__init__(**kwargs)

        self.units_full = units_full
        self.init_size = init_size
        self.num_filters = num_filters

        # Construct a list of the layers used in this block
        self.dense_layers = [
            tf.keras.layers.Dense(units_full, **actlay_config),
            tf.keras.layers.Dense(init_size*num_filters[0], **actlay_config)]

        self.deconv_layers = []
        for filters in num_filters[1:]:
            self.deconv_layers.append(
                Conv1DTranspose(filters=filters, **deconvlay_config))

        last_config = copy.deepcopy(deconvlay_config)
        last_config['strides'] = 1
        last_config['activation']= None
        self.deconv_layers.append(Conv1DTranspose(filters=1, **last_config))
        self.deconv_layers.append(tf.keras.layers.Flatten())
        
        # The boolean for adding the initial and final layers together
        self.add_init_fin = add_init_fin

    def call(self, input_tensor):
        x = input_tensor
        for layer in self.dense_layers:
            x = layer(x)
        x = tf.reshape(x, shape=(-1, self.init_size, self.num_filters[0]))
        for layer in self.deconv_layers:
            x = layer(x)
        if self.add_init_fin:
            x += input_tensor
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "init_size": self.init_size,
                "units_full": self.units_full,
                "num_filters": self.num_filters,
                "deconvlay_config": self.deconvlay_config,
                "actlay_config": self.actlay_config,
                "layers": self.layers,
                "add_init_fin": self.add_init_fin}

class Conv1DTranspose(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='valid', **kwargs):
        super().__init__()
        self.conv2dtranspose = tf.keras.layers.Conv2DTranspose(
          filters, (kernel_size, 1), (strides, 1), padding, **kwargs
        )

    def call(self, x):
        x = tf.expand_dims(x, axis=2)
        x = self.conv2dtranspose(x)
        x = tf.squeeze(x, axis=2)
        return x
