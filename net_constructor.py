# net_constructor.py
# Helper function for building networks


from tensorflow import keras
import tensorflow as tf
import numpy as np

from AbstractArchitecture_v2 import AbstractArchitecture
from DenseEncoder import DenseEncoder
from NormalizedMeanSquaredError import NormalizedMeanSquaredError as NMSE



def construct_fc_net(n, encoder_layers, decoder_layers, act_layer, lin_layer, add_identity):

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