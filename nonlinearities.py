import numpy as np
import scipy.io
import scipy.special
import math

def sigmoid_deriv(x):
    y = scipy.special.expit(x)
    return y * (1 - y)

class Nonlinearity():
    def apply(self, x):
        pass

    def gradient(self, x):
        pass
 
class Sigmoid(Nonlinearity):
    def apply(self, x):
        return scipy.special.expit(x)

    def gradient(self, x):
        return sigmoid_deriv(x)

# class Softmax(Nonlinearity):
    # def apply(self, x):
        # x = np.exp(x - np.max(x, axis=1))
        # sums = np.sum(x, axis=1)
        # return x / sums

    # def gradient(self, x):
        # # take averaged values for each to get errors
        # x = np.sum(x, axis=0) / x.shape[0]
        # # softmax jacobian
        # # i = j => p_i - p_i^2
        # # i != j => -p_i*p_j
        # jac = np.diagflat(x) - x.dot(x.T)
        # return jac
