import tensorflow as tf
from tensorflow.linalg import LinearOperatorToeplitz


def return_sdg_matrix(size):
    '''Symmetric Decreasing Gaussian matrix'''
    rows = tf.random.normal(shape=[size[0]])
    rows = tf.abs(rows)
    rows = tf.sort(rows, direction='DESCENDING')
    cols = [0.0]*size[0]
    cols[0] = rows[0].numpy()
    cols, rows

    operator_1 = LinearOperatorToeplitz(cols, rows)
    operator_1 = operator_1.to_dense()

    utm = tf.linalg.band_part(operator_1, 0, -1, name="utm")
    L = tf.multiply(0.5, utm+tf.transpose(utm), name="op")

    return L
