### Script for constructing model (this can be changed later.)

from tensorflow import keras
import tensorflow as tf
import numpy as np

from OperatorLayer import SymmetricOperator, SolveOperatorInverse
from NormalizedMeanSquaredError import NormalizedMeanSquaredError as NMSE
from DenseEncoder import DenseEncoder
from DenseDecoder import DenseDecoder


class AbstractArchitecture(keras.Model):
    def __init__(self,
                 units_full=128,
                 units_latent=20,
                 u_encoder_block=DenseEncoder(),
                 u_decoder_block=DenseDecoder(),
                 F_encoder_block=DenseEncoder(),
                 F_decoder_block=DenseDecoder(),
                 train_autoencoders_only=False,
                 latent_config=dict(activation=None),
                 **kwargs):
        super().__init__(**kwargs)  # handles standard args (e.g., name)

        # Place configuration as attributes
        self.l = units_latent

        # u autoencoder
        self.u_encoder = u_encoder_block
        self.u_Reduce = tf.Variable(self.I_seed(units_full, units_latent),
                                    trainable=True)
        self.u_Expand = tf.Variable(self.I_seed(units_latent, units_full),
                                    trainable=True)
        #self.u_latentspace = keras.layers.Dense(units_latent, **latent_config)
        self.u_decoder = u_decoder_block

        # F autoencoder
        self.F_encoder = F_encoder_block
        self.F_Reduce = tf.Variable(self.I_seed(units_full, units_latent),
                                    trainable=True)
        self.F_Expand = tf.Variable(self.I_seed(units_latent, units_full),
                                    trainable=True)
        #self.F_latentspace = keras.layers.Dense(units_latent, **latent_config)
        self.F_decoder = F_decoder_block

        # Now create the operator layer that connects v->f
        self.Operator = tf.Variable(self.I_seed(units_latent, units_latent),
                             trainable=True)

        # Set the NMSE loss function used for custom losses
        self.NMSE = NMSE(name="NormalizedMSE")

        # Boolean for whether or not to train Autoencoders only
        self.train_autoencoders_only = train_autoencoders_only

    def call(self, inputs):
        u_input, F_input = inputs

        # First, we deal with the u autoencoder
        u_encoded = self.u_encoder(u_input)
        v = tf.matmul(u_encoded, self.u_Reduce)
        v_expand = tf.matmul(v, self.u_Expand)
        u_decoded = self.u_decoder(v_expand)

        # Now the same for the f autoencoder
        F_encoded = self.F_encoder(F_input)
        f = tf.matmul(F_encoded, self.F_Reduce)
        f_expand = tf.matmul(f, self.F_Expand)
        F_decoded = self.F_decoder(f_expand)

        # Calculate symmetric L matrix
        utm = tf.linalg.band_part(self.Operator, 0, -1, name="L_upper")
        L = tf.multiply(0.5, utm+tf.transpose(utm), name="L")

        # Compute Lv(=f), then put through F_decoder
        Lv = tf.matmul(v, L)
        Lv_expand = tf.matmul(Lv, self.F_Expand)
        Lv_decoded = self.F_decoder(Lv_expand)

        # And L^-1f(= v), then put through u_decoder
        f_T = tf.transpose(f)
        Linvf_T = tf.linalg.solve(L, f_T, adjoint=True)
        Linvf = tf.transpose(Linvf_T)
        Linvf_expand = tf.matmul(Linvf, self.u_Expand)
        Linvf_decoded = self.u_decoder(Linvf_expand)

        # Now determine what to return...
        if self.train_autoencoders_only:
            zeros = 0*u_decoded
            return u_decoded, F_decoded, zeros, zeros

        # Attempting modification of the sums for superposition
        f_sums = tf.reshape(f[None]+f[:, None], [-1, f.shape[-1]])
        Lv_sums = tf.reshape(Lv[None]+Lv[:, None], [-1, Lv.shape[-1]])

        # Add the superposition loss function
        self.add_loss(self.NMSE(f_sums, Lv_sums))
        # Add loss function for Lv = f
        self.add_loss(self.NMSE(f, Lv))
        # And return the outputs

        return u_decoded, F_decoded, Lv_decoded, Linvf_decoded

    def I_seed(self, n_rows, n_cols):
        if n_rows >= n_cols:
            A = np.zeros((n_rows, n_cols), dtype=np.float32)
            for col in range(n_cols):
                for row in range(col, col+n_rows-n_cols+1):
                    A[row, col] = 1.0/(n_rows-n_cols+1)
        else:
            A = np.zeros((n_rows, n_cols), dtype=np.float32)
            for row in range(n_rows):
                for col in range(row, row+n_cols-n_rows+1):
                    A[row, col] = 1.0/(n_cols-n_rows+1)

        return A