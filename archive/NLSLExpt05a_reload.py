#!/usr/bin/python3.6
import random as r
import json

from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from AbstractArchitecture_v2 import AbstractArchitecture
from DenseEncoder import DenseEncoder
from NormalizedMeanSquaredError import NormalizedMeanSquaredError as NMSE



# Set Experiment Specifics
expt_name = "NLSL_Experiment_05a"
data_file_prefix = './data/NLSL_expt1'  ## FILL IN HERE (from file name)

# Network architecture design
l = 128  # Latent space size

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

# Batch size for model training
batch_size = 64

# Time to train autoencoders only and full models for initial seeding test
num_init_models = 20  # number of models to try with diff. learning rates

# This number is used to compute number of epochs for full-model training
final_model_train_hrs = 5


############################################
### Everything below here is automated!! ###
############################################

# Step 0. Assign a random number generator seed
x = r.randint(0, 10**(10))
r.seed(x)

# Step 1. Load in the data
data_train_u = np.load("{}_train1_u.npy".format(data_file_prefix))
data_train_f = np.load("{}_train1_f.npy".format(data_file_prefix))
data_val_u = np.load("{}_val_u.npy".format(data_file_prefix))
data_val_f = np.load("{}_val_f.npy".format(data_file_prefix))

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
                       'u_decoder_block': DenseEncoder(**decoder_config),
                       'F_encoder_block': DenseEncoder(**encoder_config),
                       'F_decoder_block': DenseEncoder(**decoder_config),
                       'latent_config': latent_config}

###############################################
## Step 3. Train 20 initial models, autoencoders-only then full model
###############################################
# create a variety of different models with randomized learning rates
models = []

# Set the loss functions
loss_fns = 4*[NMSE()]

# Set up validation data for autoencoders-only and full model training
val_zeros = np.zeros(data_val_u.shape)
val_data_aec = ([data_val_u, data_val_f], 
                [data_val_u, data_val_f, val_zeros, val_zeros])
val_data_full = ([data_val_u, data_val_f], 
                 [data_val_u, data_val_f, data_val_f, data_val_u])

    
# Create a model
full_model = AbstractArchitecture(**architecture_config,
                                  train_autoencoders_only=False)

json_file = '.model_weights/{}_initial_train.json'.format(expt_name)

with open('./model_weights/NLSL_Experiment_05a_initial_train.json') as f:
    initial_train = json.load(f)
    
lr = initial_train['best_lr']
index = initial_train['learn_rates'].index(lr)

checkpoint_model_path = './model_weights/{}_checkpoint_{}'.format(expt_name,index)
full_model.load_weights(checkpoint_model_path)

# Compile model
full_model.compile(loss=loss_fns,
                   optimizer=optimizer(learning_rate=lr, **optimizer_opts))

# Set up validation data, loss functions, and number of epochs
val_data = ([data_val_u, data_val_f], 
            [data_val_u, data_val_f, data_val_f, data_val_u])
loss_fns = 4*[NMSE()]

# Compute number of epochs to fit full model
# about 1 epoch/sec in this step
final_epochs = int(final_model_train_hrs*60*60) # 1 epoch/sec

# Continue training the (now full) model
checkpoint_model_path = './model_weights/{}_checkpoint_full'.format(expt_name)
cbs = [keras.callbacks.ModelCheckpoint(checkpoint_model_path,
                                       save_weights_only=True,
                                       monitor='val_loss',
                                       save_best_only=True)]
hist = full_model.fit(x=[data_train_u, data_train_f],
                      y=[data_train_u, data_train_f, data_train_f, data_train_u],
                      validation_data=val_data,
                      callbacks=cbs,
                      batch_size=batch_size,
                      epochs=final_epochs)

# Load weights with best validation loss
full_model.load_weights(checkpoint_model_path)

# Evaluate model at checkpoint
best_val = full_model.evaluate(x=[data_val_u, data_val_f],
                               y=[data_val_u, data_val_f, data_val_f, data_val_u],
                               verbose=False)
print("Best Validation Error: ", best_val)

# And save the final results!
full_model_path = "./model_weights/{}_final_model_weights".format(expt_name)
full_model.save(full_model_path)

## Doubled down on JSON for saving the data, since it is a uniform format!!

# Get the dictionary containing each metric and the loss for each epoch
history_dict = hist.history.copy()
    
# And now dump it
hist_filepath = "./model_weights/{}_model_history.json".format(expt_name)
json.dump(history_dict, open(hist_filepath, 'w'))


