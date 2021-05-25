# DeepGreen

This is a code repository for the manuscript ["DeepGreen: Deep Learning of Green’s Functions for Nonlinear Boundary Value Problems"](https://arxiv.org/abs/2101.07206). The repository is organized with the following directories: 
- Architecture: Tensorflow model components for the DeepGreen architecture.
- Data: Data used for training the models, including scripts for generating the raw data. Some data files are too large for GitHub and are hosted elsewhere (see README in Data directory)
- Experiments: Experiment script files that run the experiments reported in the manuscript.
- Results: Saved trained models from each of the experiments reported in the manuscript.
- Base directory: contains figure plotting notebooks

## Base Directory
The base directory contains (i) Jupyter Notebooks that recreate the figures seen in the manuscript and (ii) supporting Python files for plotting functions and analysis. Each notebook is labeled using the figure number in the manuscript for clarity. A generic analysis notebook is provided as 'Analysis Notebook.ipynb' which provides examples of how to generate all of the different plots for a given experiment. The two module files Experiment.py and figure_functions.py contain an Experiment class and plotting method definitions, respectively. The Experiment class provides simplified access to many of the interesting parts of the results from a loaded, trained model. 

## Architecture
The architecture directory contains files related to defining the TensorFlow model, loss functions, and matrix/layer initialization methods. The "Dense" files are abstracted encoder and decoder structures that use dense neural network layers for 1D and 2D data. Similarly, the "Conv" files use convolutional layers with a defined pattern alternating between convolutional and pooling layers. The "NormalizedMeanSquaredError" file specifies a TensorFlow loss function that computed a normalized mean squared error as defined in the manuscript. The module sdg_matrix provides an initialization strategy for a randomized symmetric Toeplitz matrix. The "GreenNet" file provides an abstract formulation of the DeepGreen network architecture which connects two autoencoders by the latent space with a linear matrix operator. 

## Data
In this repository, the data directory contains a README which links to a Google Drive directory with the data. In practice, you will need to copy over the data file from the Google Drive to this directory, as it should ultimately contain the raw data for training the models. The Google Drive also contains scripts for generating and organizing the data.

## Experiments
This directory contains links to experiment script files that were used to generate the results in the manuscript, including the appendix results. The general flow of an experiment file is to define the experiment name, the data to be used in the experiment, architecture details (e.g. activation functions, initializers, and regularization for each layer), and experiment details (e.g. number of epochs for autoencoder-only and full-architecture training). After setting these details, the experiment is launched using the "run_experiment" method which is defined in the utils.py module (also in the Experiments directory).

## Results
In this repository, the results directory contains a README which links to a Google Drive directory with the saved, trained models. Similar to the data directory, you will need to copy over the trained models from the Google Drive to this directory, as it should ultimately contain the results for each of the trained models.
