import tensorflow as tf
import copy


class Conv2DEncoder(tf.keras.Model):
    """A Convolutional layer-based encoder."""
    def __init__(self, 
                 num_filters = [8, 16, 32, 64], 
                 convlay_config=dict(kernel_size=4, strides=1, padding='SAME',
                                     activation='relu',
                                     kernel_initializer='he_normal'),
                 poollay_config=dict(pool_size=2, strides=2, padding='VALID'),
                 add_init_fin=True, **kwargs):
        super().__init__(**kwargs)

        # Construct a list of the convolutional and pooling layers used in this block
        self.conv_layers = [tf.keras.layers.Conv2D(filters=num_filters[0],
                                                   **convlay_config)]
        for filters in num_filters[1:]:
            self.conv_layers.append(tf.keras.layers.AveragePooling2D(**poollay_config))
            self.conv_layers.append(tf.keras.layers.Conv2D(filters=filters, 
                                                           **convlay_config))
        
        # Construct a list of the dense layers in this block
        self.proj_layer = tf.keras.layers.Conv2D(filters=num_filters[-1],
                                                 kernel_size=2**(len(num_filters)-1),
                                                 strides=2**(len(num_filters)-1))

        # The boolean for adding the initial and final layers together
        self.add_init_fin = add_init_fin

    def call(self, input_tensor):
        x = tf.expand_dims(input_tensor, axis=-1)
        for layer in self.conv_layers:
            x = layer(x)
        if self.add_init_fin:
            proj = self.proj_layer(tf.expand_dims(input_tensor, axis=-1))
            x += proj
        x = tf.keras.layers.Flatten()(x)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "num_filters": self.num_layers,
                "convlay_config": self.convlay_config,
                "poollay_config": self.poollay_config,
                "layers": self.layers,
                "add_init_fin": self.add_init_fin}

class Conv2DDecoder(tf.keras.Model):
    """A Convolutional layer-based encoder."""
    def __init__(self, 
                 init_size = [16, 16, 64],
                 output_size = (-1, 128, 128),
                 num_filters = [32, 16, 8], 
                 deconvlay_config=dict(kernel_size=4, strides=2, padding='SAME',
                                       activation='relu',
                                       kernel_initializer='he_normal'),
                 add_init_fin=True, **kwargs):
        super().__init__(**kwargs)

        self.init_size = init_size
        self.output_size = output_size

        # Construct a list of the convolutional layers used in this block
        self.deconv_layers = [
            tf.keras.layers.Conv2DTranspose(filters=filters, **deconvlay_config) 
            for filters in num_filters]

        last_config = copy.deepcopy(deconvlay_config)
        last_config['strides'] = 1
        self.deconv_layers.append(tf.keras.layers.Conv2DTranspose(filters=1, 
                                                                  **last_config))
        
        # The boolean for adding the initial and final layers together
        self.add_init_fin = add_init_fin

    def call(self, input_tensor):
        x = tf.reshape(input_tensor, 
                       shape=(-1, self.init_size[0], self.init_size[1], self.init_size[2]))
        for layer in self.deconv_layers:
            x = layer(x)
        x = tf.reshape(x, shape=self.output_size)
        if self.add_init_fin:
            proj = tf.reshape(input_tensor, shape=self.output_size)
            x += proj
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "init_size": self.init_size,
                "output_size": self.output_size,
                "num_filters": self.num_filters,
                "deconvlay_config": self.deconvlay_config,
                "layers": self.layers,
                "add_init_fin": self.add_init_fin}

