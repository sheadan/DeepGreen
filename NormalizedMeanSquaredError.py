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

    def __init__(self, norm_ord='euclidean', norm_opts={}, **kwargs):
        self.norm_ord = norm_ord
        self.norm_opts = norm_opts
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        mse = tf.square(y_pred - y_true)
        err = mse/tf.norm(y_true, ord=self.norm_ord, **self.norm_opts)
        return err

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "norm_ord": self.norm_ord,
                "norm_opts": self.norm_opts}
