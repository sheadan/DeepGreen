import random as r
import json

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.ticker as mtick
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch

from architecture.NormalizedMeanSquaredError import NormalizedMeanSquaredError as NMSE


def rel_mse(pred, true, den_nonzero=1e-5):
    num = np.mean(np.square(pred-true), axis=-1)
    den = np.mean(np.square(true), axis=-1)
    den += den_nonzero
    return np.divide(num, den) 


def compute_losses(full_model, x, y):
    predicted_ys = full_model.predict(x=x)
    data = []
    for (pred_y, true_y) in zip(predicted_ys, y):
        data.append(rel_mse(pred_y, true_y, 1e-5))
    return data


def compute_rolling_average(x, window=6):
    if window % 2 != 0:
        raise ValueError("Window must be even.")
        return 1
    # Prepare a copy of the data
    x = x.copy().flatten()
    
    # Create an empty array for storing shifted copies of the data
    mean_array = np.ndarray((x.shape[0], int(window)))

    for i in range(int(window)):
        mean_array[:,i] = np.roll(x, i)

    half_window = int(window/2)
    roll_avg = np.mean(mean_array, axis=1)
    roll_avg = roll_avg[half_window:]
    
    return roll_avg


def prediction_compare_plot(experiment, dataset_name):
    fig, [ax1, ax2] = plt.subplots(2,3, sharex=True)
    
    modes = ['Best', 'Mean', 'Worst']

    # Set the line styles for the different lines
    true_line_u = dict(color='cornflowerblue', lw=1)
    true_line_f = dict(color='orange', lw=1)
    pred_line = dict(linewidth = 0.75, linestyle='--', color='k', alpha=0.8)
    
    for j in range(3):
        mode = modes[j]
        ax_u = ax1[j]
        ax_f = ax2[j]

        #mode = 'worst'
        i = experiment.find_sample(dataset_name, mode)

        # Get a prediction
        u, F = experiment.access_data(dataset_name, int(i))
        pred_u, pred_F = experiment.predict_uF(u, F)

        # Set up x vector for plotting
        x = np.linspace(0,2*np.pi, u.shape[-1])
    
        # Plot the u's
        ax_u.plot(x, u.T, **true_line_u)
        ax_u.plot(x, pred_u.T, **pred_line)

        # Plot the f's
        ax_f.plot(x, F.T, **true_line_f)
        ax_f.plot(x, pred_F.T, **pred_line)

    # Format titles
    for j in range(3):
        ax1[j].set_title(modes[j])
    
    # Format y-axes
    ax1[0].set_ylabel(r"$\mathbf{u}(x)$")
    ax2[0].set_ylabel(r"$\mathbf{F}(x)$")
    for ax_f in ax2:
        ax_f.set_xlabel(r"$\mathbf{x}$")
    
    # Set x-axis ticks
    for j in range(3):
        ax2[j].set_xticks([0, np.pi, 2*np.pi])
        ax2[j].set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
        ax2[j].set_xlim([0,2*np.pi])
    
    true_u = ax_u.get_lines()[0]
    true_f = ax_f.get_lines()[0]
    pred_line = ax_f.get_lines()[1]
    fig.legend((true_u, true_f, pred_line), 
               ("True $\mathbf{u}(x)$", "True $\mathbf{F}(x)$", "Predicted $\mathbf{u}(x)$, $\mathbf{F}(x)$"), 
               loc='lower center', 
               bbox_to_anchor=(0.5,-0.1),
               ncol=3)
    
    plt.tight_layout()
    

def latent_space_plot(expt, index, dataset_name='train1'):
    # Set up the plotting vectors
    v, f = expt.encode_vf(index=index, dataset_name='train1')
    x = np.linspace(0, 2*np.pi, f.shape[0])

    # Some plotting options
    aec_line = dict(color='orange', marker='o', mfc='orange', mec='black')
    aec_line2 = dict(color='cornflowerblue', marker='d', mfc='cornflowerblue', mec='black')

    # And now plot it!
    fig = plt.figure()

    #Plot the v and f vectors
    plt.plot(x, v, **aec_line, label=r'$\boldsymbol{\psi}_u \mathbf{u}(\mathbf{x})$')
    plt.plot(x, f, **aec_line2, label=r'$\boldsymbol{\psi}_F \mathbf{F}(\mathbf{x})$')

    #Place the legend and label the axes
    plt.legend(loc='lower left')
    plt.ylabel(r"$\mathbf{f}(\mathbf{x})$, $\mathbf{v}(\mathbf{x})$")
    plt.xlabel(r"$\mathbf{x}$")

    # Format x-axis
    plt.xlabel(r"$\mathbf{x}$")
    plt.xticks([0, np.pi, 2*np.pi], ["0", r"$\pi$", r"$2\pi$"])
    plt.xlim([0,2*np.pi])

def loss_boxplot(expt, dataset_name='test1'):
    # Generate loss data, and set labels
    losses = expt.compute_losses(dataset_name)
    labels = [r"$\mathcal{L}_1$", r"$\mathcal{L}_2$",  r"$\mathcal{L}_5$", r"$\mathcal{L}_6$"]

    # Set up figure, axes
    fig = plt.figure()

    # Plot the losses, give the labels
    plt.boxplot(losses, labels=labels, showfliers=False, medianprops={'color': 'black'})

    # Format the Y-Axis
    plt.ylabel("Loss Function Value")
    plt.ylim([1e-6, 1e-2])
    plt.yscale('log')
    ax = plt.gca()
    ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    ax.set_yticklabels([r"$10^{-6}$", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$"])


def training_loss_epochs_plot(expt, roll_window=20):
    # Load up the training history for a given experiment
    # Cast as numpy arrays
    train_loss, val_loss = expt.get_training_losses()

    # Compute the rolling averages and determine the corresponding epoch indices for plotting
    roll_window = 20
    train_roll = compute_rolling_average(train_loss, window=roll_window)
    val_roll = compute_rolling_average(val_loss, window=roll_window)
    roll_idcs = np.arange(train_loss.shape[0])[int(roll_window/4):-int(roll_window/4)]

    #Set the figure
    plt.figure()

    # Plot the actual epoch-by-epoch losses
    plt.semilogy(val_loss, label="Validation Loss", alpha=0.75)
    plt.semilogy(train_loss, label="Training Loss", alpha=0.75)

    # And rolling averages
    plt.semilogy(roll_idcs, val_roll, color='royalblue')#, label="Validation Loss (rolling average)")
    plt.semilogy(roll_idcs, train_roll, color='brown')#, label="Training Loss (rolling average)")

    # Set up a vertical line indicating where the Operator L enabled
    plt.axvline(75, linestyle='-.', color='k', alpha=0.9)#, linewidth=0.7)
    ax = plt.gca()
    arrow = FancyArrowPatch((250,2), (80,2), fc='k', arrowstyle='Simple', mutation_scale=5)
    ax.add_patch(arrow)
    annotate_text = r"$\mathcal{L}_3$-$\mathcal{L}_6$ activated"
    ax.annotate(annotate_text, xy=(250,1.65), fontsize=8)
    #r"Operator $L$ enabled"

    # Format the axes
    plt.xlabel("Epochs")
    plt.ylabel("Cumulative Loss")
    plt.legend(loc='upper right')
    plt.xlim([0,val_loss.shape[0]])
