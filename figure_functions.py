import random as r
import json

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.ticker as mtick
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection

from architecture.NormalizedMeanSquaredError import NormalizedMeanSquaredError as NMSE


def set_axes(ax):
    if ax is not None:
        # If axes are provided, set current axes:
        plt.sca(ax)
    else:
        # Otherwise instantiate a figure
        plt.figure()


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


def prediction_compare_plot(experiment, dataset_name, fig=None, axs=None, label=True):
    if fig is None or axs is None:
        fig, [ax1, ax2] = plt.subplots(2,3, sharex=True)
    else:
        [ax1, ax2] = axs
    
    modes = ['Best', 'Mean', 'Worst']

    # Set the line styles for the different lines
    true_line_u = dict(color='orange')
    true_line_f = dict(color='cornflowerblue')
    pred_line = dict(linestyle='--', color='k', alpha=0.8)
    
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
    if label:
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
               bbox_to_anchor=(0.5,-0.05),
               ncol=3)
    
    plt.tight_layout()
    

def latent_space_plot(expt, index, dataset_name='train1', ax=None):
    # Create a figure or instantiate new axes:
    set_axes(ax)

    # Set up the plotting vectors
    v, f = expt.encode_vf(index=index, dataset_name='train1')
    x = np.linspace(0, 2*np.pi, f.shape[0])

    # Some plotting options
    aec_line = dict(color='orange', marker='o', mfc='orange', mec='black')
    aec_line2 = dict(color='cornflowerblue', marker='d', mfc='cornflowerblue', mec='black')

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

def loss_boxplot(expt, dataset_name='test1', ax=None):
    # Generate loss data, and set labels
    losses = expt.compute_losses(dataset_name)
    labels = [r"$\mathcal{L}_1$", r"$\mathcal{L}_2$", r"$\mathcal{L}_3$",  r"$\mathcal{L}_5$", r"$\mathcal{L}_6$"]

    # Create a figure or instantiate new axes:
    set_axes(ax)

    # Plot the losses, give the labels
    plt.boxplot(losses, labels=labels, showfliers=False, medianprops={'color': 'black'})

    # Format the Y-Axis
    plt.ylabel("Normalized MSE")
    plt.ylim([1e-6, 1e-3])
    plt.yscale('log')
    ax = plt.gca()
    ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3])
    ax.set_yticklabels([r"$10^{-6}$", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$"])


def training_loss_epochs_plot(expt, roll_window=20, ax=None):
    # Load up the training history for a given experiment
    # Cast as numpy arrays
    train_loss, val_loss = expt.get_training_losses()

    # Compute the rolling averages and determine the corresponding epoch indices for plotting
    roll_window = 20
    train_roll = compute_rolling_average(train_loss, window=roll_window)
    val_roll = compute_rolling_average(val_loss, window=roll_window)
    roll_idcs = np.arange(train_loss.shape[0])[int(roll_window/4):-int(roll_window/4)]

    # Create a figure or instantiate new axes:
    set_axes(ax)

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
    plt.ylabel("Normalized MSE")
    plt.legend(loc='upper right')
    plt.xlim([0,val_loss.shape[0]])



def waterfall_plot(fig, ax, X, Y, Z, **kwargs):
    '''
    Make a waterfall plot
    Input:
        fig,ax : matplotlib figure and axes to populate
        Z : n,m numpy array. Must be a 2d array even if only one line should be plotted
        X,Y : n,m array
        kwargs : kwargs are directly passed to the LineCollection object
    '''
    # Set normalization to the same values for all plots
    norm = plt.Normalize(Z.min().min(), Z.max().max())
    
    # Check sizes to loop always over the smallest dimension
    n,m = Z.shape
    if n>m:
        X=X.T; Y=Y.T; Z=Z.T
        m,n = n,m

    for j in range(n):
        # reshape the X,Z into pairs 
        points = np.array([X[j,:], Z[j,:]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)  
        # The values used by the colormap are the input to the array parameter
        lc = LineCollection(segments, norm=norm, array=(Z[j,1:]+Z[j,:-1])/2, **kwargs)
        line = ax.add_collection3d(lc,zs=(Y[j,1:]+Y[j,:-1])/2, zdir='y') # add line to axes

    ax.auto_scale_xyz(X,Y,Z) # set axis limits
    
def generate_GL_plot(expt, G_off=0.7, L_off=0.7, fig=None, axs=None):
    # Grab the relevant L and G matrices
    L = np.array(expt.L)
    G = expt.G

    # Set up and plot
    x = np.linspace(0,2*np.pi, L.shape[0])
    X,XI = np.meshgrid(x,x)

    # Options
    cmap = 'terrain'
    waterfall_opts = dict(linewidth=1.5, alpha=1.0, cmap=cmap)
    contour_opts = dict(levels=100, zdir='z', cmap=cmap)

    # Set limits for z-axis
    Gm = np.min(G)
    GM = np.max(G)
    Goffset = Gm - G_off * (GM - Gm)
    Lm = np.min(L)
    LM = np.max(L)
    Loffset = Lm - L_off * (LM - Lm)

    if fig is None or axs is None:
        # set up the axes for the first plot, plot G
        fig = plt.figure(figsize=(7.3,3.65))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    else:
        ax1 = axs[0]
        ax2 = axs[1]

    # First plot of G:
    waterfall_plot(fig, ax1, X, XI, G, **waterfall_opts) 
    ax1.contourf(X, XI, G, offset=Goffset, vmin=Gm, vmax=GM, **contour_opts)

    # Format x-axis
    ax1.set_xlabel(r"$\mathbf{x}$", labelpad=-5)
    ax1.set_xticks([0, np.pi, 2*np.pi])
    ax1.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax1.set_xlim([0,2*np.pi])

    # Format xi-axis
    ax1.set_ylabel(r'$\boldsymbol{\xi}$', labelpad=-5)
    ax1.set_yticks([0, np.pi, 2*np.pi])
    ax1.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax1.set_ylim([0,2*np.pi])

    # Format z-axis
    ax1.set_zlim(Goffset, GM)

    # Adjust the paddings
    ax1.xaxis.set_rotate_label(False)
    ax1.yaxis.set_rotate_label(False)
    ax1.tick_params(axis='both', pad=-0.5)
    ax1.tick_params(axis='x', pad=-5)
    ax1.tick_params(axis='y', pad=-5)

    # Place the z-axis label using text command
    ax1.text(x=-0.5, y=2*np.pi, z=1.2*np.max(G), s=r'$\mathbf{G}$')#rotation_mode=None, rotation=180)

    # Set up the view position angle
    ax1.view_init(15,215)

    # Now plot of L
    waterfall_plot(fig, ax2, X, XI, L, **waterfall_opts) 
    ax2.contourf(X, XI, L, offset=Loffset, vmin=Lm, vmax=LM, **contour_opts)

    # Format x-axis
    ax2.set_xlabel(r"$\mathbf{x}$", labelpad=-5)
    ax2.set_xticks([0, np.pi, 2*np.pi])
    ax2.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax2.set_xlim([0,2*np.pi])

    # Format xi-axis
    ax2.set_ylabel(r'$\boldsymbol{\xi}$', labelpad=-5)
    ax2.set_yticks([0, np.pi, 2*np.pi])
    ax2.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax2.set_ylim([0,2*np.pi])

    # Format z-axis
    ax2.set_zlim(Loffset, LM)

    # Adjust the paddings
    ax2.xaxis.set_rotate_label(False)
    ax2.yaxis.set_rotate_label(False)
    ax2.tick_params(axis='both', pad=-2)
    ax2.tick_params(axis='x', pad=-5)
    ax2.tick_params(axis='y', pad=-5)

    # Place the z-axis label using text command
    ax2.text(x=-0.5, y=2*np.pi, z=1.2*np.max(L), s=r'$\mathbf{L}$')#rotation_mode=None, rotation=180)

    # Set up the view position angle
    ax2.view_init(15,215)


def generate_G_plot(expt, G_off=0.7):
    # Grab the relevant L and G matrices
    G = expt.G

    # Set up and plot
    x = np.linspace(0,2*np.pi, G.shape[0])
    X,XI = np.meshgrid(x,x)

    # Plotting options
    cmap = 'terrain'
    waterfall_opts = dict(linewidth=1.5, alpha=1.0, cmap=cmap)
    contour_opts = dict(levels=100, zdir='z', cmap=cmap)

    # Initialize a figure
    fig = plt.figure()
    
    # Set limits for z-axis
    Gm = np.min(expt.G)
    GM = np.max(expt.G)
    Goffset = Gm - G_off * (GM - Gm)

    # set up the axes for the first plot, plot G
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    waterfall_plot(fig, ax, X, XI, expt.G, **waterfall_opts) 
    ax.contourf(X, XI, expt.G, offset=Goffset, vmin=Gm, vmax=GM, **contour_opts)

    # Format x-axis
    ax.set_xlabel(r"$\mathbf{x}$", labelpad=-5)
    ax.set_xticks([0, np.pi, 2*np.pi])
    ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlim([0,2*np.pi])

    # Format xi-axis
    ax.set_ylabel(r'$\boldsymbol{\xi}$', labelpad=-5)
    ax.set_yticks([0, np.pi, 2*np.pi])
    ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax.set_ylim([0,2*np.pi])
    
    # Adjust the paddings
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.tick_params(axis='both', pad=-2)
    ax.tick_params(axis='x', pad=-5)
    ax.tick_params(axis='y', pad=-5)

    # Format z-axis
    ax.set_zlim(Goffset, GM)

    # Place the z-axis label using text command
    ax.text(x=-0.5, y=2*np.pi, z=1.4*np.max(expt.G), s=r'$\mathbf{G}(x,\xi)$')#rotation_mode=None, rotation=180)

    # Set up the view position angle
    ax.view_init(15,215)
    
    # Return the figure reference
    return fig

def summary_boxplot(s0, s1, s2):
    # Generate loss data, and set labels
    s0_losses = s0.compute_losses('test1')
    s1_losses = s1.compute_losses('test1')
    s2_losses = s2.compute_losses('test1')
    labels = [r"$\mathcal{L}_1$", r"$\mathcal{L}_2$", r"$\mathcal{L}_3$",  r"$\mathcal{L}_5$", r"$\mathcal{L}_6$"]

    # Set up figure, axes
    fig = plt.figure()

    # Plot each series
    b0=plt.boxplot(s0_losses,
                   positions=[0.75, 1.75, 2.75, 3.75, 4.75],
                   medianprops={'color': 'black'},
                   showfliers=False, widths=0.2)

    b1=plt.boxplot(s1_losses,
                   positions=[1, 2, 3, 4, 5],
                   medianprops={'color': 'green'},
                   showfliers=False, widths=0.2)

    b2=plt.boxplot(s2_losses,
                   positions=[1.25, 2.25, 3.25, 4.25, 5.25],
                   medianprops={'color': 'purple'},
                   showfliers=False, widths=0.2)

    # Format the X axis
    plt.xticks([1,2,3,4, 5], labels)

    # Format the Y-Axis
    plt.ylabel("Normalized MSE")
    plt.ylim([1e-6, 1e-1])
    plt.yscale('log')
    ax = plt.gca()
    ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_yticklabels([r"$10^{-6}$", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"])

    # Format the legend
    plt.legend([b0['medians'][0], b1['medians'][0], b2['medians'][0]],
               ["System 1", "System 2", "System 3"],
               ncol=3, loc='center', bbox_to_anchor=(0.5, -0.28))

