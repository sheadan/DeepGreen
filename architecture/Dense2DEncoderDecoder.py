import tensorflow as tf
import math

class Dense2DEncoder(tf.keras.Model):
    """A Dense layer-based encoder."""
    def __init__(self, units_full=128, 
                 units_hidden=128, num_layers=3,
                 actlay_config=dict(activation="elu",
                                    kernel_initializer='he_normal'),
                 linlay_config=dict(activation=None),
                 add_init_fin=True, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.units_full = units_full

        # Construct a list of the layers used in this block
        self.block_layers = []
        # The first (num_layers-1) layers will be dense, activated layers
        for i in range(num_layers-1):
            self.block_layers.append(tf.keras.layers.Dense(units_hidden,
                                                           **actlay_config))
        # The final layer does not have activation
        self.block_layers.append(tf.keras.layers.Dense(units_full,
                                                       **linlay_config))
        # The boolean for adding the initial and final layers together
        self.add_init_fin = add_init_fin

    def call(self, input_tensor):
        x = input_tensor
        x = tf.reshape(x, shape=(-1, self.units_full))
        for layer in self.block_layers:
            x = layer(x)
        if self.add_init_fin:
            x += tf.reshape(input_tensor, shape=(-1, self.units_full))
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "num_layers": self.num_layers,
                "actlay_config": self.actlay_config,
                "linlay_config": self.linlay_config,
                "units_full": self.units_full,
                "layers": self.layers,
                "add_layer": self.add_layer,
                "add_init_fin": self.add_init_fin}


class Dense2DDecoder(tf.keras.Model):
    """A Dense layer-based encoder."""
    def __init__(self, units_full=128, 
                 units_hidden=128, num_layers=3,
                 actlay_config=dict(activation="elu",
                                    kernel_initializer='he_normal'),
                 linlay_config=dict(activation=None),
                 add_init_fin=True, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.units_full = units_full

        # Construct a list of the layers used in this block
        self.block_layers = []
        # The first (num_layers-1) layers will be dense, activated layers
        for i in range(num_layers-1):
            self.block_layers.append(tf.keras.layers.Dense(units_hidden,
                                                           **actlay_config))
        # The final layer does not have activation
        self.block_layers.append(tf.keras.layers.Dense(units_full,
                                                       **linlay_config))
        # The boolean for adding the initial and final layers together
        self.add_init_fin = add_init_fin

    def call(self, input_tensor):
        x = input_tensor
        for layer in self.block_layers:
            x = layer(x)
        if self.add_init_fin:
            x += input_tensor
        x = tf.reshape(x, shape=(-1, math.isqrt(self.units_full), math.isqrt(self.units_full)))
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "num_layers": self.num_layers,
                "actlay_config": self.actlay_config,
                "linlay_config": self.linlay_config,
                "units_full": self.units_full,
                "layers": self.layers,
                "add_layer": self.add_layer,
                "add_init_fin": self.add_init_fin}

