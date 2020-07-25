### Script for constructing model (this can be changed later.)

from tensorflow import keras
import tensorflow as tf
import numpy as np

from OperatorLayer import SymmetricOperator, SolveOperatorInverse
from EncoderDecoder import DenseEncoder, DenseDecoder

def Architecture(units_full=128, 
                 units_latent=20, 
                 u_encoder=DenseEncoder(name='u_encoder'),
                 u_decoder=DenseDecoder(name='u_decoder'),
                 F_encoder=DenseEncoder(name='F_encoder'),
                 F_decoder=DenseDecoder(name='F_decoder'),
                 latent_config=dict(activation=None, use_bias=False),
                 **kwargs):
    
    # Network inputs
    input_u = keras.layers.Input(shape=[units_full], name='input_u')
    input_F = keras.layers.Input(shape=[units_full], name='input_F')
    
    # First, we deal with the u autoencoder
    u_encoded = u_encoder(input_u)
    u_Reduce = keras.layers.Dense(units_latent, 
                                  kernel_initializer=I_seed,
                                  name='u_Reduce', 
                                  **latent_config)
    v = u_Reduce(u_encoded)
    u_Expand = keras.layers.Dense(units_full, 
                                  kernel_initializer=I_seed,
                                  name='u_Expand', 
                                  **latent_config)
    v_expand = u_Expand(v)
    u_decoded = u_decoder(v_expand)

    # Now the same for the f autoencoder
    F_encoded = F_encoder(input_F)
    F_Reduce = keras.layers.Dense(units_latent, 
                                  kernel_initializer=I_seed,
                                  name='F_Reduce', 
                                  **latent_config)
    f = F_Reduce(F_encoded)
    F_Expand = keras.layers.Dense(units_full, 
                                  kernel_initializer=I_seed,
                                  name='F_Expand', 
                                  **latent_config)
    f_expand = F_Expand(f)
    F_decoded = F_decoder(f_expand)

    # Compute Lv(=f), then put through F_decoder
    Operator = SymmetricOperator()
    Lv = Operator(v)
    #Lv = keras.layers.Dense(units_latent)(v)
    Lv_expand = F_Expand(Lv)
    Lv_decoded = F_decoder(Lv_expand)

    # And L^-1f(= v), then put through u_decoder
    Inverse = SolveOperatorInverse(Operator)
    Linvf = Inverse(f)
    #Linvf = keras.layers.Dense(units_latent)(f)
    Linvf_expand = u_Expand(Linvf)
    Linvf_decoded = u_decoder(Linvf_expand)

    # Autoencoders only model
    autoencoders = keras.Model(inputs=[input_u, input_F], 
                               outputs=[u_decoded, F_decoded],
                               name='autoencoders')
    # Full model
    model = keras.Model(inputs=[input_u, input_F], 
                        outputs=[u_decoded, F_decoded, Lv_decoded, Linvf_decoded],
                        name='full_model')

    # Sums for superposition
    f_sums = tf.reshape(f[None]+f[:, None], [-1, f.shape[-1]])
    Lv_sums = tf.reshape(Lv[None]+Lv[:, None], [-1, Lv.shape[-1]])

    # Add the superposition loss function
    model.add_loss(NMSE(f_sums, Lv_sums))
    # Add loss function for Lv = f
    model.add_loss(NMSE(f, Lv))
    
    return autoencoders, model

def I_seed(shape, dtype=tf.float32):
    n_rows = shape[0]
    n_cols = shape[1]
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

    return tf.constant(A, dtype=dtype)

def NMSE(y_true, y_pred, denom_nonzero=1e-5):
    mse = tf.reduce_mean(tf.square(y_pred-y_true), axis=-1)
    true_norm = tf.reduce_mean(tf.square(y_true), axis=-1)
    true_norm += denom_nonzero
    err = tf.truediv(mse, true_norm)
    
    return tf.reduce_mean(err)

