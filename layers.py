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
        self._cached_inputs = None
        self._cached_activations = None

    def forward(self, inputs, cache=False):
        linear = inputs.dot(self._weights) + self._biases

        # apply nonlinearity
        if self._nonlinearity is not None:
            activations = self._nonlinearity.apply(linear)
        else:
            activations = linear

        # cache these for backpropagation
        if cache:
            self._cached_sum = linear
            self._cached_inputs = prev_activations
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
        grad_wrt_weights = self._cached_inputs.T.dot(grad_wrt_sum)
        grad_wrt_biases = grad_wrt_sum.sum(axis=0)

        # calculate error wrt inputs, 1 row for each example
        in_errors = grad_wrt_sum.dot(self._weights.T)

        compensated_lr = lr / batch_size
        self._weights -= grad_wrt_weights * compensated_lr
        self._biases -= grad_wrt_biases * compensated_lr
         
        return in_errors

class Recurrent(Layer):

    def __init__(self, input_size, output_size, init_weights_size, nonlinearity):
        self._weights_input = np.array(np.random.rand(input_size, output_size) - 0.5) * init_weights_size
        self._weights_hidden = np.array(np.random.rand(output_size, output_size) - 0.5) * init_weights_size
        self._biases = np.zeros(output_size)

        self._nonlinearity = nonlinearity
        self._hidden_states = []

    def forward(self, inputs, cache=False):
        if len(self._hidden_states) == 0:
            # seed with 0 vectors of size n x output
            last_state = self.zeros((len(inputs), len(self.weights_output[0])))
        else:
            last_state = self._hidden_states[-1]

        h_t = self.inputs.dot(self._weights_input) + last_state.dot(self._weights_output) + self.biases
        
        if self.cache:
            self._hidden_states.append(h_t)
        else:
            self._hidden_states[-1] = h_t

    def backprop(self, out_errors, lr):
        '''
        out errors - tuple (t, errors) dL/dh_t 
        '''
        
        d_w_input = np.zeros(self._weights_input.shape)
        d_w_hidden = np.zeros(self._weights_hidden.shape)

        for e in out_errors:
            for i in reversed(range(t + 1)):
                g = self._nonlinearity.gradient(self._hidden_states[i])


