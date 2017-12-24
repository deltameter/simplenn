import numpy as np

class Layer():
    def forward(self, examples):
        pass
    def backprop(self):
        pass

class FullyConnected(Layer):
    def __init__(self, input_size, output_size, init_weights_size, nonlinearity):
        self._weights = np.array(np.random.rand(input_size, output_size) - 0.5) * init_weights_size
        self._biases = np.zeros(output_size)
        self._nonlinearity = nonlinearity
        self._cached_prev_activations = None
        self._cached_activations = None

    def forward(self, prev_activations, cache=False):
        linear = prev_activations.dot(self._weights) + self._biases

        # apply nonlinearity
        if self._nonlinearity is not None:
            activations = self._nonlinearity.apply(linear)
        else:
            activations = linear

        # cache these for backpropagation
        if cache:
            self._cached_sum = linear
            self._cached_prev_activations = prev_activations
            self._cached_activations = activations

        return activations

    def backprop(self, out_errors, lr):
        ''' input: n x #out features array of partial derivatives w.r.t the output of this layer 
        output: n x #in features array of partial derivatives w.r.t. output last layer ''' 
        batch_size = self._cached_activations.shape[0]
        # z_ij = g(0_1 * z_1_j-1 + ... 0_n * y_n_j-1 + b)
        # dn/d0 = g'(...) * z_i_j-1
        # so calculate the quantity g'(...) first and multiply by dL/dn
        if self._nonlinearity is not None:
            grad_wrt_sum = self._nonlinearity.gradient(self._cached_sum) * out_errors
        else:
            grad_wrt_sum = out_errors
        # calculate error wrt weights
        # get gradient by accumulating gradient for each training ex.
        grad_wrt_weights = self._cached_prev_activations.T.dot(grad_wrt_sum)
        grad_wrt_biases = grad_wrt_sum.sum(axis=0)

        # calculate error wrt prev output, 1 row for each example
        in_errors = grad_wrt_sum.dot(self._weights.T)

        compensated_lr = lr / batch_size
        self._weights -= grad_wrt_weights * compensated_lr
        self._biases -= grad_wrt_biases * compensated_lr
         
        return in_errors
