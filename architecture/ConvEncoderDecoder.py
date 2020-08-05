import tensorflow as tf


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
