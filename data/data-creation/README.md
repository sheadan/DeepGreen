# Creating data for DeepGreen

This code was used to create the data sets in ["DeepGreen: Deep Learning of Green's Function for Nonlinear Boundary Value Problems"](https://arxiv.org/abs/2101.07206) by Craig R. Gin, Daniel E. Shea, Steven L. Brunton, and J. Nathan Kutz. 

The one-dimensional data sets were created using MATLAB and the two-dimensional data sets were created using Python. Instructions for each are found below. 

## To create one-dimensional data

### 1. Create the forcing functions

Run `ForcingGenerator.m`. This creates a file called `bvp_forcings_1.0.mat` that contains the cosine, Gaussian, and cubic polynomial forcing functions. 

### 2. Solve the boundary value problems

There are three scripts, each of which solves one of the one-dimensional systems in the paper using MATLAB's built in bvp5c. `BVP5cMassSolverS0.m`  solves the cubic Helmholtz equation using the forcing functions in `bvp_forcings_1.0.mat` and saves the solutions in a file named `Computed_Solutions-S0-datetime.mat`. Similarly, the nonlinear Sturm-Liouville equation is solved using `BVP5cMassSolverS1.m` and the nonlinear biharmonic equation is solved using `BVP5cMassSolverS2.m`.

### 3. Find acceptable solutions

The bvp5c method may not converge for all systems and forcing functions. The script `BVP5cSaveResults.m` will load the output .mat files from the previous step, get rid of solutions that fail to meet some error tolerance, and save the results in an organized manner. This script must be run separately for each of the three one-dimensional data sets. You must set the variable *forcing_file* to the name of the file `Computed_Solutions...` from the previous step. The results will be saved to a file whose name is specified by the variable *output_file*. 

### 4. Split data into train/validation/test and convert to numpy arrays

Each data set needs to be split into training data, validation data, and test data. Furthermore, the test data is split into data with forcing functions that are similar to the training data (cosine and Gaussian) and data with forcing functions that are dissimilar to the training data (cubic polynomial). Then these data sets are saved as numpy arrays so they can easily be loaded for training with the Tensorflow models. This is done with the Python script `DataSplit.py`. This script must be run separately for each of the three one-dimensional data sets. Currently, it loads the cubic Helmholtz data from the file `S0data.mat` and saves the output files with the prefix `S0-Oscillator_`. These two strings can be changed in order to split the other two data sets.

## To create two-dimensional data

The two-dimensional data set was created with Python using a finite element method implemented with the DOLFIN library of the FEniCS Project. For installation instructions, see https://fenicsproject.org/olddocs/dolfin/latest/python/installation.html.

### 1. Solve the boundary value problem

Run the script `Nonlinear_Poisson_Cosine.py` to solve the nonlinear Poisson equation with cosine forcing functions. The output will be two files named `S3-NLP_Cosine_us.npy` and `S3-NLP_Cosine_fs.npy` which contain the solutions and forcing functions, respectively. Similarly, the scripts `Nonlinear_Poisson_Gaussian.py` and `Nonlinear_Poisson_Polynomial.py` solve the nonlinear Poisson equation with Gaussian and cubic polynomial forcing functions and save the results.

### 2. Split data into train/validation/test 

The data set is split into training, validation, and test data using the script `DataSplit-2D.py`.
