import numpy as np
import scipy.io
import scipy.special
import math
from losses import *
from nonlinearities import *
from utils import *

class network():
    def __init__(self, loss, lr=0.3):
        self._layers = []
        self._loss = loss
        self._lr = lr

    def add_layer(self, layer):
        self._layers.append(layer)

    def infer(self, examples):
        return self._forwardprop(examples)

    def _forwardprop(self, examples, cache=False):
        activations = examples

        for i in range(len(self._layers)):
            activations = self._layers[i].forward(activations, cache=cache)

        return activations
    # the hard part lmao

    def _backprop(self, examples, labels, lr):
        output = self._forwardprop(examples, cache=True)
        
        loss = self._loss.objective(output, labels)
           
        for i in list(range(len(self._layers)))[::-1]:
            loss = self._layers[i].backprop(loss, lr)

    def train(self, data, batch_size, epochs, lr):
        if lr is None:
            lr = self._lr

        examples, labels = data

        for _ in range(epochs):
            # randomize the order
            random_permutation = np.random.permutation(examples.shape[0])
            examples = examples[random_permutation]
            labels = labels[random_permutation]

            # perform minibatch SGD
            for start in range(0, len(examples), batch_size):
                end = start + batch_size
                self._backprop(examples[start:end], labels[start:end], lr)
                
            self._display_accuracy(examples[:5000], labels[:5000])

        print('Training finished.')

    def _display_accuracy(self, examples, labels):
        error = calc_accuracy(encode_to_onehot(self.infer(examples)), labels)
        print('Training error of {:2}'.format(error)) 
        return error
