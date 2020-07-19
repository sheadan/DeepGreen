#!/usr/bin/python3
import random as r
import json

from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from AbstractArchitecture_v2 import AbstractArchitecture
from DenseEncoder import DenseEncoder
from DenseDecoder import DenseDecoder
from NormalizedMeanSquaredError import NormalizedMeanSquaredError as NMSE


# Inputs to score model
model_weight_path = './CheckpointTest.tmp'
model_weight_path = './model_weights/NLSL_Experiment_03c_final_model_weights.tf'
l = 128  # Latent space size
data_file_prefix = './data/NLSL_expt1'

# Reconstruct network:
activation = "relu"
initializer = tf.keras.initializers.VarianceScaling()
reg_lambda_l2 = 1e-6
regularizer = tf.keras.regularizers.l2(reg_lambda_l2)

act_layer = dict(activation=activation,
                 kernel_initializer=initializer,
                 kernel_regularizer=regularizer)
lin_layer = dict(activation=None,
                 kernel_initializer=initializer,
                 kernel_regularizer=regularizer)
latent_config = dict(activation=None,
                     kernel_regularizer=regularizer,
                     use_bias=False)

encoder_layers = 5
decoder_layers = 5
add_identity = True

# Model training setting
## Set optimizer
optimizer = keras.optimizers.Adam
optimizer_opts = {}

# Callback function(s) and fit method options
cbs = []

############################################
### Everything below here is automated!! ###
############################################
# Step 1. Load in the data
data_train_u = np.load("{}_train1_u.npy".format(data_file_prefix)).astype(np.float32)
data_train_f = np.load("{}_train1_f.npy".format(data_file_prefix)).astype(np.float32)
data_val_u = np.load("{}_val_u.npy".format(data_file_prefix)).astype(np.float32)
data_val_f = np.load("{}_val_f.npy".format(data_file_prefix)).astype(np.float32)
data_test_u1 = np.load("{}_test1_u.npy".format(data_file_prefix)).astype(np.float32)
data_test_f1 = np.load("{}_test1_f.npy".format(data_file_prefix)).astype(np.float32)
data_test_u = np.load("{}_test2_u.npy".format(data_file_prefix)).astype(np.float32)
data_test_f = np.load("{}_test2_f.npy".format(data_file_prefix)).astype(np.float32)

# Step 2. Set up the model architecture
_, n = data_train_u.shape

encoder_config = {'units_full': n,
                  'num_layers': encoder_layers,
                  'actlay_config': act_layer,
                  'linlay_config': lin_layer,
                  'add_init_fin': add_identity}

decoder_config = {'units_full': n,
                  'num_layers': decoder_layers,
                  'actlay_config': act_layer,
                  'linlay_config': lin_layer,
                  'add_init_fin': add_identity}

# Aggregate settings for model architecture
architecture_config = {'units_latent': l,
                       'units_full': n,
                       'u_encoder_block': DenseEncoder(**encoder_config),
                       'u_decoder_block': DenseDecoder(**decoder_config),
                       'F_encoder_block': DenseEncoder(**encoder_config),
                       'F_decoder_block': DenseDecoder(**decoder_config),
                       'latent_config': latent_config}

###############################################
# Step 5. Set up the full architecture run!!
###############################################

# Set up validation data, loss functions, and number of epochs
val_data = [(data_val_u, data_val_f),
            (data_val_u, data_val_f, data_val_f, data_val_u)]
loss_fns = 4*[NMSE()]

# Instantiate the new model
full_model = AbstractArchitecture(**architecture_config,
                                  train_autoencoders_only=False)

# Load the weights
full_model.load_weights(model_weight_path)
full_model.compile(loss=loss_fns, optimizer=optimizer())

# Predicted y values
#predicted_ys = full_model.predict(x=[data_train_u, data_train_f])

full_model.test_on_batch(x=[data_train_u, data_train_f],
                         y=[data_train_u, data_train_f, data_train_f, data_train_u])
