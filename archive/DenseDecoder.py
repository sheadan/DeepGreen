import tensorflow as tf


class DenseDecoder(tf.keras.Model):
    """A Dense layer-based encoder."""
    def __init__(self, units_full=128, num_layers=4,
                 actlay_config=dict(activation="elu",
                                    kernel_initializer='he_normal'),
                 linlay_config=dict(activation=None),
                 add_init_fin=True, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        # The first layer is important for the identity add, so separate it
        self.first_layer = tf.keras.layers.Dense(units_full, **linlay_config)
        # Construct a list of the layers used in this block
        self.block_layers = []
        # All layers will be dense, activated layers
        for i in range(num_layers-1):
            self.block_layers.append(tf.keras.layers.Dense(units_full,
                                                           **actlay_config))
        self.block_layers.append(tf.keras.layers.Dense(units_full, **linlay_config))

        # Boolean for adding the initial and final layers
        self.add_init_fin = add_init_fin


    def call(self, input_tensor):
        x = self.first_layer(input_tensor)
        for layer in self.block_layers:
            x = layer(x)
        if self.add_init_fin:
            x += self.first_layer(input_tensor)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "num_layers": self.num_layers,
                "actlay_config": self.actlay_config,
                "units_full": self.units_full,
                "layers": self.layers,
                "add_layer": self.add_layer,
                "add_init_fin": self.add_init_fin}

