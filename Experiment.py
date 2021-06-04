'''Experiment object for collecting data related to an experiment.'''

import os, json

import numpy as np
import tensorflow as tf
from tensorflow.keras import models

from figure_functions import rel_mse
from architecture.NormalizedMeanSquaredError import NormalizedMeanSquaredError as NMSE


class Experiment:
    def __init__(self, experiment_name:str, 
                 data_file_prefix: str, 
                 results_folder = 'results',
                 data_folder = 'data'):
        # Save the experiment name
        self.name = experiment_name

        # Load in the experiment model
        model_path = [os.getcwd(), results_folder, experiment_name, "final_model"]
        model_path = os.path.sep.join(model_path)
        custom_objects = {"NormalizedMeanSquaredError": NMSE}
        self.model = models.load_model(model_path, custom_objects=custom_objects)

        # Load the experiment data
        datasets = {'train1_u', 'train1_f', 'val_u', 'val_f', 
                    'test1_u', 'test1_f', 'test2_u', 'test2_f'}
        self.data = {}
        data_path_parts = [os.getcwd(), data_folder, ""]
        for dset in datasets:
            data_path_parts[-1] = data_file_prefix + "_" + dset
            path = os.path.sep.join(data_path_parts) + ".npy"
            self.data[dset] = np.load(path).astype(np.float32)

        # Load the training history
        hist_path = [os.getcwd(), results_folder, experiment_name, "initial_pool_results.json"]
        dict_path = os.path.sep.join(hist_path)
        self.init_train_hist = json.load(open(dict_path))
        hist_path[-1] = "final_model_history.json"
        dict_path = os.path.sep.join(hist_path)
        self.final_train_hist = json.load(open(dict_path))

        # Set G and L
        utm = tf.linalg.band_part(self.model.Operator, 0, -1, name="L_upper")
        self.L = tf.multiply(0.5, utm+tf.transpose(utm), name="L")
        self.G = np.linalg.inv(self.L)

        # And grab the autoencoder components phi and psi
        self.u_enc = self.model.layers[0]
        self.F_enc = self.model.layers[2]

    def access_data(self, dataset_name='train1', index=None):
        # Access the appropriate data set
        u = self.data[dataset_name+"_u"]
        F = self.data[dataset_name+"_f"]
        # If an index is specified, grab that specific entry
        if type(index)== int:
            u = u[index,:].reshape(1,-1)
            F = F[index,:].reshape(1,-1)
        # Return the matrices
        return u, F

    def predict(self, index: int, dataset_name='train1'):
        # grab the relevant data:
        u, F = self.access_data(dataset_name, index)
        # Return the predicted vectors
        return self.predict_uF(u, F)

    def predict_uF(self, u, F):
        # Predict u and F given F and u, respectively
        _, _, pred_F, pred_u = self.model.predict(x=[u, F])
        return pred_u, pred_F

    def encode_vf(self, index: int, dataset_name='train1'):
        # grab the relevant data:
        u, F = self.access_data(dataset_name, index)
        # Project it into the latent space
        v = tf.matmul(self.u_enc(u), self.model.u_Reduce)
        f = tf.matmul(self.F_enc(F), self.model.F_Reduce)
        # Return the latent space vectors
        return v.numpy().flatten(), f.numpy().flatten()


    def find_sample(self, dataset_name='train1', mode='worst', meanidx=None, index=None):
        # grab the relevant data:
        u, F = self.access_data(dataset_name, index)

        # Predict `f given u` and `u given f`
        pred_u, pred_F = self.predict_uF(u, F)

        # Compute f prediction and u prediction scores
        f_scores = rel_mse(pred_F, F)
        u_scores = rel_mse(pred_u, u)
        
        # Add scores (cumulative score)
        score = np.abs(f_scores) + np.abs(u_scores)
        
        # Compute index for the given mode
        if mode.lower() == 'worst':
            idc = np.argmax(score)
        elif mode.lower() == 'best':
            idc = np.argmin(score)
        elif mode.lower() == 'mean':
            mean = np.mean(score)
            if meanidx:
                idcs = np.where(np.abs(score-mean) < 0.01*np.std(score))[0]
                idc = idcs[meanidx]
            else:
                idc = np.argmin((score-mean)**2)
            #print(idc)
        else:
            print('{} mode not supported.'.format(mode))
            idc = None
            
        return int(idc)

    def compute_losses(self, dataset_name='train1'):
        # grab the relevant data:
        u, F = self.access_data(dataset_name)
        # Prepare data to be scored:
        true_ys = [u, F, F, u]
        predicted_ys = self.model.predict(x=[u, F])
        # Compute the losses
        losses = []
        for (pred_y, true_y) in zip(predicted_ys, true_ys):
            losses.append(rel_mse(pred_y, true_y, 1e-5))
        # Since the losses are actually returned as L2, L1, L5, L6 we will swap
        # the positions of first two such that the list returns L1, L2, L5, L6
        losses[0], losses[1] = losses[1], losses[0]
        # Now we'll compute the linearity loss L3 Lv=f:
        u, F = self.access_data(dataset_name)
        # Project it into the latent space
        v = tf.matmul(self.u_enc(u), self.model.u_Reduce)
        f = tf.matmul(self.F_enc(F), self.model.F_Reduce)
        # Compute L[v]
        Lv = tf.matmul(v, self.L)
        # Determine loss
        lin_loss = rel_mse(Lv, f)
        # Insert the linear loss into the list:
        losses.insert(2, lin_loss)
        #print('Linearity Loss:', lin_loss)
        # Compute superposition loss L4
        f_sums = tf.reshape(f[None]+f[:, None], [-1, f.shape[-1]])
        Lv_sums = tf.reshape(Lv[None]+Lv[:, None], [-1, Lv.shape[-1]])
        super_loss = rel_mse(Lv_sums, f_sums)
        # Insert the superposition loss into the list:
        losses.insert(3, super_loss)
        return losses

    def evaluate_model(self, dataset_name):
        u, F = self.access_data(dataset_name)
        losses = self.model.evaluate(x=[u,F], y=[u,F,F,u])
        return losses

    def get_training_losses(self):
        # Determine the best model index number in the initial results dictionary
        best_model_idc = np.argmin(self.init_train_hist['best_loss'])

        # String together the training and validation losses
        train_loss = self.init_train_hist['aec_hist'][best_model_idc]['loss'] \
                     + self.init_train_hist['full_hist'][best_model_idc]['loss'] \
                     + self.final_train_hist['loss']
        val_loss = self.init_train_hist['aec_hist'][best_model_idc]['val_loss'] \
                   + self.init_train_hist['full_hist'][best_model_idc]['val_loss'] \
                   + self.final_train_hist['val_loss']

        # return the loss vectors
        return np.asarray(train_loss), np.asarray(val_loss)

from architecture.NormalizedMeanSquaredError import NormalizedMeanSquaredError2D as NMSE2

class Experiment2D:
    def __init__(self, experiment_name:str, 
                 data_file_prefix: str, 
                 results_folder = 'results',
                 data_folder = 'data'):
        # Save the experiment name
        self.name = experiment_name

        # Load in the experiment model
        model_path = [os.getcwd(), results_folder, experiment_name, "final_model"]
        model_path = os.path.sep.join(model_path)
        custom_objects = {"NormalizedMeanSquaredError2D": NMSE2}
        self.model = models.load_model(model_path, custom_objects=custom_objects)

        # Load the experiment data
        datasets = {'train1_u', 'train1_f', 'val_u', 'val_f', 
                    'test1_u', 'test1_f', 'test2_u', 'test2_f'}
        self.data = {}
        data_path_parts = [os.getcwd(), data_folder, ""]
        for dset in datasets:
            data_path_parts[-1] = data_file_prefix + "_" + dset
            path = os.path.sep.join(data_path_parts) + ".npy"
            self.data[dset] = np.load(path).astype(np.float32)

        # Load the training history
        hist_path = [os.getcwd(), results_folder, experiment_name, "initial_pool_results.json"]
        dict_path = os.path.sep.join(hist_path)
        self.init_train_hist = json.load(open(dict_path))
        hist_path[-1] = "final_model_history.json"
        dict_path = os.path.sep.join(hist_path)
        self.final_train_hist = json.load(open(dict_path))

        # Set G and L
        utm = tf.linalg.band_part(self.model.Operator, 0, -1, name="L_upper")
        self.L = tf.multiply(0.5, utm+tf.transpose(utm), name="L")
        self.G = np.linalg.inv(self.L)

        # And grab the autoencoder components phi and psi
        self.u_enc = self.model.layers[0]
        self.F_enc = self.model.layers[2]

    def access_data(self, dataset_name='train1', index=None):
        # Access the appropriate data set
        u = self.data[dataset_name+"_u"]
        F = self.data[dataset_name+"_f"]
        # If an index is specified, grab that specific entry
        if type(index)== int:
            u = u[index:index+1,:,:]
            F = F[index:index+1,:,:]
        # Return the matrices
        return u, F

    def predict(self, index: int, dataset_name='train1'):
        # grab the relevant data:
        u, F = self.access_data(dataset_name, index)
        # Return the predicted vectors
        return self.predict_uF(u, F)

    def predict_uF(self, u, F):
        # Predict u and F given F and u, respectively
        _, _, pred_F, pred_u = self.model.predict(x=[u, F])
        return pred_u, pred_F

    def encode_vf(self, index: int, dataset_name='train1'):
        # grab the relevant data:
        u, F = self.access_data(dataset_name, index)
        # Project it into the latent space
        v = tf.matmul(self.u_enc(u), self.model.u_Reduce)
        f = tf.matmul(self.F_enc(F), self.model.F_Reduce)
        # Return the latent space vectors
        return v.numpy().flatten(), f.numpy().flatten()


    def find_sample(self, dataset_name='train1', mode='worst', meanidx=None, index=None):
        # grab the relevant data:
        u, F = self.access_data(dataset_name, index)

        # Predict `f given u` and `u given f`
        pred_u, pred_F = self.predict_uF(u, F)

        # Compute f prediction and u prediction scores
        f_scores = self.rel_mse(pred_F, F)
        u_scores = self.rel_mse(pred_u, u)
        
        # Add scores (cumulative score)
        score = np.abs(f_scores) + np.abs(u_scores)
        
        # Compute index for the given mode
        if mode.lower() == 'worst':
            idc = np.argmax(score)
        elif mode.lower() == 'best':
            idc = np.argmin(score)
        elif mode.lower() == 'mean':
            mean = np.mean(score)
            if meanidx:
                idcs = np.where(np.abs(score-mean) < 0.01*np.std(score))[0]
                idc = idcs[meanidx]
            else:
                idc = np.argmin((score-mean)**2)
            #print(idc)
        else:
            print('{} mode not supported.'.format(mode))
            idc = None
            
        return int(idc)

    def compute_losses(self, dataset_name='train1'):
        # grab the relevant data:
        u, F = self.access_data(dataset_name)
        # Prepare data to be scored:
        true_ys = [u, F, F, u]
        predicted_ys = self.model.predict(x=[u, F])
        losses = []
        for (pred_y, true_y) in zip(predicted_ys, true_ys):
            losses.append(self.rel_mse(pred_y, true_y, 1e-5))
        # Since the losses are actually returned as L2, L1, L5, L6 we will swap
        # the positions of first two such that the list returns L1, L2, L5, L6
        losses[0], losses[1] = losses[1], losses[0]
        # Now we'll compute the linearity loss L3 Lv=f:
        u, F = self.access_data(dataset_name)
        # Project it into the latent space
        v = tf.matmul(self.u_enc(u), self.model.u_Reduce)
        f = tf.matmul(self.F_enc(F), self.model.F_Reduce)
        # Compute L[v]
        Lv = tf.matmul(v, self.L)
        # Determine loss
        lin_loss = self.rel_mse(Lv, f)
        # Insert the linear loss into the list:
        losses.insert(2, lin_loss)
        # Compute superposition loss L4
        f_sums = tf.reshape(f[None]+f[:, None], [-1, f.shape[-1]])
        Lv_sums = tf.reshape(Lv[None]+Lv[:, None], [-1, Lv.shape[-1]])
        super_loss = self.rel_mse(Lv_sums, f_sums)
        #print('Linearity Loss:', lin_loss)
        return losses

    def evaluate_model(self, dataset_name):
        u, F = self.access_data(dataset_name)
        losses = self.model.evaluate(x=[u,F], y=[u,F,F,u])
        return losses

    def get_training_losses(self):
        # Determine the best model index number in the initial results dictionary
        best_model_idc = np.argmin(self.init_train_hist['best_loss'])

        # String together the training and validation losses
        train_loss = self.init_train_hist['aec_hist'][best_model_idc]['loss'] \
                     + self.init_train_hist['full_hist'][best_model_idc]['loss'] \
                     + self.final_train_hist['loss']
        val_loss = self.init_train_hist['aec_hist'][best_model_idc]['val_loss'] \
                   + self.init_train_hist['full_hist'][best_model_idc]['val_loss'] \
                   + self.final_train_hist['val_loss']

        # return the loss vectors
        return np.asarray(train_loss), np.asarray(val_loss)

    def rel_mse(self, pred, true, den_nonzero=1e-5):
        # New rel_mse for 2D
        num = np.mean(np.mean(np.square(pred-true), axis=-1), axis=-1)
        den = np.mean(np.mean(np.square(true), axis=-1), axis=-1)
        den += den_nonzero
        return np.divide(num, den) 