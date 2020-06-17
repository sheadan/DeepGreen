### Script for constructing model (this can be changed later.)

from tensorflow import keras
import tensorflow as tf

from OperatorLayer import SymmetricOperator, SolveOperatorInverse
from NormalizedMeanSquaredError import NormalizedMeanSquaredError as NMSE
from DenseEncoder import DenseEncoder
from DenseDecoder import DenseDecoder


class AbstractArchitecture(keras.Model):
    def __init__(self, units_latent=20,
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
        self.u_latentspace = keras.layers.Dense(units_latent, **latent_config)
        self.u_decoder = u_decoder_block

        # F autoencoder
        self.F_encoder = F_encoder_block
        self.F_latentspace = keras.layers.Dense(units_latent, **latent_config)
        self.F_decoder = F_decoder_block

        # Now create the operator layer that connects v->f
        self.Operator = SymmetricOperator()

        # and its inverse f->v
        self.Inverse = SolveOperatorInverse(self.Operator)

        # Set the NMSE loss function used for custom losses
        self.NMSE = NMSE(name="NormalizedMSE")

        # Boolean for whether or not to train Autoencoders only
        self.train_autoencoders_only = train_autoencoders_only


    def call(self, inputs):
        u_input, F_input = inputs

        #print(type(F_input), F_input.shape)

        # First, we deal with the u autoencoder
        u_encoded = self.u_encoder(u_input)
        v = self.u_latentspace(u_encoded)
        u_decoded = self.u_decoder(v)

        # Now the same for the f autoencoder
        F_encoded = self.F_encoder(F_input)
        f = self.F_latentspace(F_encoded)
        F_decoded = self.F_decoder(f)

        # Compute Lv(=f), then put through F_decoder
        Lv = self.Operator(v)
        Lv_decoded = self.F_decoder(Lv)

        # And L^-1f(= v), then put through u_decoder
        Linvf = self.Inverse(f)
        Linvf_decoded = self.u_decoder(Linvf)

        # Solve the inverse problem
        #sym_op = self.Operator.get_operator()
        #X_T = tf.transpose(f)
        #LinvX_T = tf.linalg.solve(sym_op, X_T, adjoint=True)
        #Linvf = tf.transpose(LinvX_T)
        #Linvf_decoded = self.u_decoder(Linvf)

        # Add the superposition loss function
        f_sums = tf.reshape(f[None]+f[:, None], [-1, self.l])
        Lv_sums = tf.reshape(Lv[None]+Lv[:, None], [-1, self.l])

        # Now determine what to return...
        if self.train_autoencoders_only:
            zeros = 0*u_decoded
            return u_decoded, F_decoded, zeros, zeros

        # Add the superposition loss function
        self.add_loss(self.NMSE(f_sums, Lv_sums))
        # Add loss function for Lv = f
        self.add_loss(self.NMSE(f, Lv))
        # And return the outputs

        return u_decoded, F_decoded, Lv_decoded, Linvf_decoded
