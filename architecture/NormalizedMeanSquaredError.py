#from tensorflow.keras.losses import LossFunctionWrapper
from tensorflow import keras
import tensorflow as tf


class NormalizedMeanSquaredError(keras.losses.Loss):
    """Computes normalized mean of squares of errors between labels and preds.

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) name for the loss.
        norm_ord: (Optional) The 'ord' parameter for backend norm method
        norm_opts: (Optional) Additional parameters for norm method

    Normalization is based on the 'true' values
    """

    def __init__(self, denom_nonzero=1e-5, **kwargs):
        self.denom_nonzero = denom_nonzero
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        # Compute the MSE and the L2 norm of the true values (with 1/batch_size prefactor)
        mse = tf.reduce_mean(tf.square(y_pred-y_true), axis=-1)
        true_norm = tf.reduce_mean(tf.square(y_true), axis=-1)
        # Ensure there are no 'zero' values in the denominator before division
        true_norm += self.denom_nonzero

        #diff_norm = tf.norm(y_pred - y_true, ord=2, axis=-1)
        #diff_norm = tf.square(diff_norm)
        #true_norm = tf.norm(y_true, ord=2, axis=-1)  
        #true_norm = tf.square(true_norm)
        #true_norm += self.denom_nonzero
        #true_norm = tf.clip_by_value(true_norm, 1e-10, 1e24)

        # Compute normalized MSE (normalized to true L2 norm)
        err = tf.truediv(mse, true_norm)

        # Return the error
        return err

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "denom_nonzero": self.denom_nonzero}

class NormalizedMeanSquaredError2D(keras.losses.Loss):
    """Computes normalized mean of squares of errors between labels and preds.

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) name for the loss.
        norm_ord: (Optional) The 'ord' parameter for backend norm method
        norm_opts: (Optional) Additional parameters for norm method

    Normalization is based on the 'true' values
    """

    def __init__(self, denom_nonzero=1e-5, **kwargs):
        self.denom_nonzero = denom_nonzero
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        # Compute the MSE and the L2 norm of the true values (with 1/batch_size prefactor)
        mse = tf.reduce_mean(tf.reduce_mean(tf.square(y_pred-y_true), axis=-1), axis=-1)
        true_norm = tf.reduce_mean(tf.reduce_mean(tf.square(y_true), axis=-1), axis=-1)
        # Ensure there are no 'zero' values in the denominator before division
        true_norm += self.denom_nonzero

        #diff_norm = tf.norm(y_pred - y_true, ord=2, axis=-1)
        #diff_norm = tf.square(diff_norm)
        #true_norm = tf.norm(y_true, ord=2, axis=-1)  
        #true_norm = tf.square(true_norm)
        #true_norm += self.denom_nonzero
        #true_norm = tf.clip_by_value(true_norm, 1e-10, 1e24)

        # Compute normalized MSE (normalized to true L2 norm)
        err = tf.truediv(mse, true_norm)

        # Return the error
        return err

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "denom_nonzero": self.denom_nonzero}