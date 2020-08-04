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
        # Generate a symmetric matrix from the upper triangular portion
        sym_op = self.get_operator()
        # Return the multiplied matrices, so output is the same size as input
        Lv = tf.matmul(X, sym_op, name="Lv")
        return Lv


    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "initializer": self.initializer}

    def get_operator(self, X=None):
        # Grab the upper triangular portion of matrix
        utm = tf.linalg.band_part(self.operator, 0, -1, name="L_upper")
        # Generate a symmetric matrix from the upper triangular portion
        return tf.multiply(0.5, utm+tf.transpose(utm), name="L")



class SolveOperatorInverse(keras.layers.Layer):
    """A layer which applies inverse of a given operator."""
    def __init__(self, operator_layer, **kwargs):
        super().__init__(**kwargs)
        self.operator_layer = operator_layer

    def call(self, X):
        # Generate a symmetric matrix from the upper triangular portion
        #try:
        sym_op = self.operator_layer.get_operator()
        #except AttributeError:
        #    sym_op = tf.eye(X.shape[-1])
        # Solve the inverse problem
        X_T = tf.transpose(X)
        LinvX_T = tf.linalg.solve(sym_op, X_T, adjoint=True)
        LinvX = tf.transpose(LinvX_T)

        return LinvX

    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "operator_layer": self.operator_layer}

