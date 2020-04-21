import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import Identity


class SymmetricOperator(keras.layers.Layer):
    """A layer which applies a symmetric operator."""
    def __init__(self, initializer=Identity(gain=1.0), **kwargs):
        super().__init__(**kwargs)
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, batch_input_shape):
        op_shape = [batch_input_shape[-1], batch_input_shape[-1]]

        self.operator = self.add_weight(name="operator",
                                        shape=op_shape,
                                        initializer=self.initializer,
                                        trainable=True)

        super().build(batch_input_shape)

    def call(self, X):
        # Grab the upper triangular portion of matrix
        utm = tf.linalg.band_part(self.operator, 0, -1, name="L_upper")
        # Generate a symmetric matrix from the upper triangular portion
        sym_op = tf.multiply(0.5, utm+tf.transpose(utm), name="L")
        # Return the multiplied matrices, so output is the same size as input
        #return X @ sym_op
        Lv = tf.matmul(X, sym_op, name="Lv")
        return Lv


    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "initializer": self.initializer}

    def get_operator(self):
        # Grab the upper triangular portion of matrix
        utm = tf.linalg.band_part(self.operator, 0, -1, name="L_upper")
        # Generate a symmetric matrix from the upper triangular portion
        return tf.multiply(0.5, utm+tf.transpose(utm), name="L")
