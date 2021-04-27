import numpy as np
from sklearn.model_selection import train_test_split

# Load data
Cosine_us = np.load('S3-NLP_Cosine_us.npy')
Cosine_fs = np.load('S3-NLP_Cosine_fs.npy')

Gaussian_us = np.load('S3-NLP_Gaussian_us.npy')
Gaussian_fs = np.load('S3-NLP_Gaussian_fs.npy')

Polynomial_us = np.load('S3-NLP_Polynomial_us.npy')
Polynomial_fs = np.load('S3-NLP_Polynomial_fs.npy')

# Split data into train, test

# Combine Cosine and Gaussians into a single matrix - leave Polynomials for test only
data_us = np.vstack((Cosine_us, Gaussian_us))
data_fs = np.vstack((Cosine_fs, Gaussian_fs))

data_train_us, data_test1_us, data_train_fs, data_test1_fs = train_test_split(data_us, data_fs, test_size=0.1)

# Further split training and validation data
data_train_us, data_val_us, data_train_fs, data_val_fs = train_test_split(data_train_us, data_train_fs, test_size=0.2)

# Save files
prefix = 'S3-NLP_'

np.save(prefix + 'train1_u', data_train_us)
np.save(prefix + 'train1_f', data_train_fs)
np.save(prefix + 'val_u', data_val_us)
np.save(prefix + 'val_f', data_val_fs)
np.save(prefix + 'test1_u', data_test1_us)
np.save(prefix + 'test1_f', data_test1_fs)
np.save(prefix + 'test2_u', Polynomial_us)
np.save(prefix + 'test2_f', Polynomial_fs)