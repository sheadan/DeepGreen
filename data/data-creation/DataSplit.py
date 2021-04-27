import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
mat = scipy.io.loadmat('S0data.mat')

Cosine_us = mat['cos_us']
Cosine_fs = mat['cos_fs']

Gaussian_us = mat['gaussian_us']
Gaussian_fs = mat['gaussian_fs']

Polynomial_us = mat['poly_us']
Polynomial_fs = mat['poly_fs']

# Split data into train, test

# Combine Cosine and Gaussians into a single matrix - leave Polynomials for test only
data_us = np.vstack((Cosine_us, Gaussian_us))
data_fs = np.vstack((Cosine_fs, Gaussian_fs))

data_train_us, data_test1_us, data_train_fs, data_test1_fs = train_test_split(data_us, data_fs, test_size=0.1)

# Further split training and validation data
data_train_us, data_val_us, data_train_fs, data_val_fs = train_test_split(data_train_us, data_train_fs, test_size=0.2)

# Save files
prefix = 'S0-Oscillator_'

np.save(prefix + 'train1_u', data_train_us)
np.save(prefix + 'train1_f', data_train_fs)
np.save(prefix + 'val_u', data_val_us)
np.save(prefix + 'val_f', data_val_fs)
np.save(prefix + 'test1_u', data_test1_us)
np.save(prefix + 'test1_f', data_test1_fs)
np.save(prefix + 'test2_u', Polynomial_us)
np.save(prefix + 'test2_f', Polynomial_fs)
