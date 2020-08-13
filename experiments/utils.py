# Experiment Helper Functions
import random as r
import json
import sys
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer

# Add the architecture path for the GreenNet and NMSE
sys.path.append("../architecture/")
from GreenNet import GreenNet
from NormalizedMeanSquaredError import NormalizedMeanSquaredError as NMSE


def construct_network(units_full: int,
                      units_latent: int,
                      encoder_block: Layer,
                      decoder_block: Layer,
                      encoder_config: dict,
                      decoder_config: dict,
                      train_autoencoders_only: bool = False,
                      ):

    # Aggregate settings for model architecture
    architecture_config = {'units_latent': units_latent,
                           'units_full': units_full,
                           'u_encoder_block': encoder_block(**encoder_config),
                           'u_decoder_block': decoder_block(**decoder_config),
                           'F_encoder_block': encoder_block(**encoder_config),
                           'F_decoder_block': decoder_block(**decoder_config),
                           'train_autoencoders_only': train_autoencoders_only}

    model = GreenNet(**architecture_config)

    return model


def get1Ddatasize(data_file_prefix: str):
    return np.load("{}_train1_u.npy".format(data_file_prefix)).shape


def get_data(data_file_prefix: str):
    # Step 1. Load in the data
    data_train_u = np.load("{}_train1_u.npy".format(data_file_prefix))
    data_train_f = np.load("{}_train1_f.npy".format(data_file_prefix))
    data_val_u = np.load("{}_val_u.npy".format(data_file_prefix))
    data_val_f = np.load("{}_val_f.npy".format(data_file_prefix))

    # Step 2. Organize the data for different training steps
    train_zeros = np.zeros(data_train_u.shape)
    val_zeros = np.zeros(data_val_u.shape)

    train_x = (data_train_u, data_train_f)
    train_y_aec = (data_train_u, data_train_f, train_zeros, train_zeros)
    train_y_full = (data_train_u, data_train_f, data_train_f, data_train_u)

    val_x = (data_val_u, data_val_f)
    val_y_aec = (data_val_u, data_val_f, val_zeros, val_zeros)
    val_y_full = (data_val_u, data_val_f, data_val_f, data_val_u)
    val_aec = (val_x, val_y_aec)
    val_full = (val_x, val_y_full)

    return train_x, train_y_aec, train_y_full, val_aec, val_full


def evaluate_initial_models(save_prefix, train_data,
                            train_opts, network_config):
    # Extract the training data
    tx, ty_aec, ty_full, v_aec, v_full = train_data

    print(type(tx), len(tx), type(tx[0]), len(tx[0]))#tx[0].shape)

    # Gather the relevant training options
    aec_only_epochs = train_opts['aec_only_epochs']
    init_full_epochs = train_opts['init_full_epochs']
    num_init_models = train_opts['num_init_models']
    loss_fn = train_opts['loss_fn']
    opt = train_opts['optimizer']
    optimizer_opts = train_opts['optimizer_opts']
    batch_size = train_opts['batch_size']

    # Set up results dictionary
    results = {'full_hist': [],
               'aec_hist': [],
               'lr': [],
               'best_loss': [],
               'model_path': []}

    # For loop for generating, training, and evaluating the initial model pool
    for i in range(num_init_models):
        # Randomly selected learning rate
        lr = 10**(-r.uniform(2, 5))

        # Create a model, initially only to train autoencoders!
        model = construct_network(train_autoencoders_only=True, **network_config)

        # Compile the model
        model.compile(loss=4*[loss_fn], optimizer=opt(lr=lr, **optimizer_opts))

        # Set up the Callback function
        checkpoint_path_aec = save_prefix + 'checkpoint_aec_{}'.format(i)
        cbs_aec = [keras.callbacks.ModelCheckpoint(checkpoint_path_aec,
                                                   save_weights_only=True,
                                                   monitor='val_loss',
                                                   save_best_only=True)]

        #  Fit autoencoder-only model
        aec_hist = model.fit(x=tx, y=ty_aec, validation_data=v_aec,
                             callbacks=cbs_aec, batch_size=batch_size,
                             epochs=aec_only_epochs, verbose=2)

        # Re-load weights with best validation loss
        model.load_weights(checkpoint_path_aec)

        # Now set the model to train all aspects (including operator L)
        model.train_autoencoders_only = False

        # Re-compile the model
        model.compile(loss=4*[loss_fn], optimizer=opt(lr=lr, **optimizer_opts))

        # Train full model
        checkpoint_path_full = save_prefix + 'checkpoint_{}'.format(i)
        cbs = [keras.callbacks.ModelCheckpoint(checkpoint_path_full,
                                               save_weights_only=True,
                                               monitor='val_loss',
                                               save_best_only=True)]

        # Fit the full model
        full_hist = model.fit(x=tx, y=ty_full, validation_data=v_full,
                              callbacks=cbs, batch_size=batch_size,
                              epochs=init_full_epochs)

        # Load weights with best validation loss
        model.load_weights(checkpoint_path_full)

        # Evaluate model at checkpoint
        best_loss = model.evaluate(x=v_full[0], y=v_full[1], verbose=False)

        # Save the model
        model_path = save_prefix + "model_{}".format(i)
        model.save(model_path)

        # Append the results to the model list
        results['full_hist'].append(full_hist.history.copy())
        results['aec_hist'].append(aec_hist.history.copy())
        results['lr'].append(lr)
        results['best_loss'].append(best_loss[0])
        results['model_path'].append(model_path)

        # Delete the model variable and clear_session to remove any graph
        del model
        tf.keras.backend.clear_session()

    # Select the best model from the loop
    best_model_idc = np.argmin(results['best_loss'])
    best_model_path = results['model_path'][best_model_idc]

    # Return the best model's path
    return results, best_model_path


def train_final_model(model_path: str, save_prefix: str,
                      train_data,
                      train_opts: dict,
                      custom_objects: dict = {}):
    # Gather the relevant training options
    best_model_epochs = train_opts['best_model_epochs']
    batch_size = train_opts['batch_size']

    # Extract the training data
    tx, ty_aec, ty_full, v_aec, v_full = train_data

    # Load the model
    model = tf.keras.models.load_model(model_path,
                                       custom_objects=custom_objects)

    # set the place to save the checkpoint model weights
    checkpoint_model_path = save_prefix + 'checkpoint_final'

    # Define the callback function for training
    cbs = [keras.callbacks.ModelCheckpoint(checkpoint_model_path,
                                           save_weights_only=True,
                                           monitor='val_loss',
                                           save_best_only=True)]

    # Extract the training data
    tx, ty_aec, ty_full, v_aec, v_full = train_data

    # Train the best model
    hist = model.fit(x=tx, y=ty_full, validation_data=v_full,
                     callbacks=cbs, batch_size=batch_size,
                     epochs=best_model_epochs)

    # Load the best model weights
    model.load_weights(checkpoint_model_path)

    # Save the best model
    model_path = save_prefix + 'final_model'
    model.save(model_path)

    return hist.history, model_path

def save_results(results_path: str, random_seed: int,
                 model_path: str, custom_objects: dict,
                 final_hist: dict, init_hist: dict):
    # Load the model
    model = tf.keras.models.load_model(model_path,
                                       custom_objects=custom_objects)
    model.save(results_path + 'final_model')
    print("Best model saved to:", model_path)

    # Export the final model training dictionary
    hist_filepath = results_path + "initial_pool_results.json"
    json.dump(init_hist, open(hist_filepath, 'w'))

    # Export the final model training dictionary
    final_hist['random_seed'] = random_seed
    hist_filepath = results_path + "final_model_history.json"
    json.dump(final_hist, open(hist_filepath, 'w'))

    print("Exported training dictionaries to: ", results_path)


def check_for_directories(expt_name: str):
    pardir = os.path.abspath(os.pardir)
    dirs = ['logs',
            'model_weights',
            'model_weights' + os.sep + expt_name,
            'results',
            'results' + os.sep + expt_name,
            ]
    for dirname in dirs:
        os.makedirs(pardir+os.sep+dirname, exist_ok=True)


def run_experiment(random_seed, expt_name: str, data_file_prefix: str,
                   training_options: dict, network_config: dict,
                   custom_objects: dict = {"NormalizedMeanSquaredError": NMSE}
                   ):
    # Assign a random number generator seed for learning rates
    r.seed(random_seed)
    check_for_directories(expt_name)

    # Get the training data
    train_data = get_data(data_file_prefix)

    # Set the prefix for where to save the results/checkpointed models
    save_prefix = '../model_weights/{}/'.format(expt_name)

    # Step 1 -- Train a collection of initial models
    # Autoencoders-only, then full model
    # This method returns the file path to the best model:
    init_hist, model_path = evaluate_initial_models(save_prefix,
                                                    train_data,
                                                    training_options,
                                                    network_config)

    # Step 2 -- Load the best model, and train for the full time
    # Load the best model (note: assumes the use of NMSE loss function)
    final_hist, model_path = train_final_model(model_path, save_prefix,
                                               train_data,
                                               training_options,
                                               custom_objects)

    # Step 3 -- Save the results
    results_path = '../results/{}/'.format(expt_name)
    save_results(results_path, random_seed,
                 model_path, custom_objects,
                 final_hist, init_hist)
