#!/usr/bin/python3
import random as r
import sys

import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices("GPU"):
  tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow import keras
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.activations import relu

from utils import run_experiment, get1Ddatasize

# Add the architecture path for the DenseEncoderDecoder and NMSE
sys.path.append("../architecture/")
from DenseEncoderDecoder import DenseEncoderDecoder
from NormalizedMeanSquaredError import NormalizedMeanSquaredError as NMSE


# Example Experiment Script:
expt_name = 'S0-22'
data_file_prefix = '../data/S0-Oscillator'

# Set size of latent space, and retrieve the 'full' size of the data
units_latent = 20
units_full = get1Ddatasize(data_file_prefix)[-1] # the last dimension is the 'length' of the data, for 1D data

# Set up encoder and decoder configuration dict(s)
activation = relu
initializer = keras.initializers.VarianceScaling()
regularizer = l1_l2(0, 1e-6)

actlay_config = {'activation': activation,
                 'kernel_initializer': initializer,
                 'kernel_regularizer': regularizer}

linlay_config = {'activation': None,
                 'kernel_initializer': initializer,
                 'kernel_regularizer': regularizer}

enc_dec_config = {'units_full': units_full,
                  'num_layers': 5,
                  'actlay_config': actlay_config,
                  'linlay_config': linlay_config,
                  'add_init_fin': True}

# Network configuration (this is how the AbstractArchitecture will be created)
network_config = {'units_full': units_full,
                  'units_latent': units_latent,
                  'u_encoder_block': DenseEncoderDecoder(**enc_dec_config),
                  'u_decoder_block': DenseEncoderDecoder(**enc_dec_config),
                  'F_encoder_block': DenseEncoderDecoder(**enc_dec_config),
                  'F_decoder_block': DenseEncoderDecoder(**enc_dec_config),
                  'operator_initializer': initializer}

# Aggregate all the training options in one dictionary
training_options = {'aec_only_epochs': 75, 
                    'init_full_epochs': 250,
                    'best_model_epochs': 2500,
                    'num_init_models': 20, 
                    'loss_fn': NMSE(),
                    'optimizer': keras.optimizers.Adam,
                    'optimizer_opts': {},
                    'batch_size': 64
                    }

####################################################################
### Launch the Experiment
####################################################################

# Get a random number generator seed
random_seed = r.randint(0, 10**(10))

# Set the custom objects used in the model (for loading purposes)
custom_objs = {"NormalizedMeanSquaredError": NMSE}

# And run the experiment!
run_experiment(random_seed=random_seed,
               expt_name=expt_name,
               data_file_prefix=data_file_prefix,
               training_options=training_options,
               network_config=network_config,
               custom_objects=custom_objs)
