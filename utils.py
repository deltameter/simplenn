import numpy as np
import scipy.io
import scipy.special
import math

def encode_to_onehot(output):
    output = np.eye(output.shape[1])[np.argmax(output, axis=1)]
    return output

def calc_accuracy(output, truth):
    assert len(output) == len(truth), 'Output and truth must be same length'
    return np.sum(np.argmax(output, 1) == np.argmax(truth, 1)) / len(output)

