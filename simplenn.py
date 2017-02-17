import numpy as np
import scipy.io
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

sigmoid = np.vectorize(sigmoid)
sigmoid_deriv = np.vectorize(sigmoid_deriv)

def encode_to_onehot(output):
    output = np.eye(output.shape[1])[list(np.argmax(output, 1))]
    return output

def calc_accuracy(output, truth):
    assert len(output) == len(truth), 'Output and truth must be same length'
    return np.sum(np.argmax(output, 1) == np.argmax(truth, 1)) / len(output)

class SimpleNN():
    # HYPER PARAMETERS
    # Each entry represents a layer of size n
    # These are just defaults, can be set train() function as well
    HIDDEN_LAYERS = []
    NUM_EPOCHS = 50
    BATCH_SIZE = 100
    LEARNING_RATE = 0.1

    # Each entry represents a layer of size n
    INPUT_SIZE = 784
    OUTPUT_LAYER_SIZE = 10

    TRAIN_DATA_LOC = "./digit-dataset/train.mat"
    TEST_DATA_LOC = "./digit-dataset/test.mat"

    def __init__(self):
        self.weights = []
        self.biases = []

        # create matrix of bias as weights
        all_layer_sizes = [self.INPUT_SIZE] + self.HIDDEN_LAYERS + [self.OUTPUT_LAYER_SIZE]

        for i in range(1, len(all_layer_sizes)):
            # randomly initialize weights to break symmetry
            weight_matrix = (np.random.rand(all_layer_sizes[i], all_layer_sizes[i-1]) - 0.5)
            bias_matrix = np.random.rand(all_layer_sizes[i], 1) / 10
            self.weights.append(weight_matrix)
            self.biases.append(bias_matrix)

        # load our dataset
        train_dataset = scipy.io.loadmat(self.TRAIN_DATA_LOC)
        test_dataset = scipy.io.loadmat(self.TEST_DATA_LOC)
        self.train_data = train_dataset['train_images']
        self.train_labels = train_dataset['train_labels']
        self.test_data = test_dataset['test_images']

        # didn't come with labels?
        # self.test_labels = test_dataset['test_labels']

        # labels to one hot
        self.train_labels = np.eye(self.OUTPUT_LAYER_SIZE)[self.train_labels.T.tolist()]

        # simple normalization
        self.train_data = (self.train_data / np.amax(self.train_data))
        self.test_data = (self.test_data / np.amax(self.test_data))
        
        # reshape data into matrix of (m, feature_size)
        self.train_data = np.swapaxes(self.train_data.reshape((self.INPUT_SIZE, self.train_data.shape[2])), 0, 1)
        self.test_data = np.swapaxes(self.test_data.reshape((self.INPUT_SIZE, self.test_data.shape[2])), 0, 1)

    def forward(self, examples):
        assert examples.shape[0] == self.INPUT_SIZE, 'Input vectors must be of size {0} not size {1}'.format(self.INPUT_SIZE, examples.shape[0])

        # save dot products and activated neurons so we don't have to do so in back prop
        dots = []
        activations = [examples]

        for i in range(len(self.weights)):
            examples = self.weights[i].dot(examples)
            examples += self.biases[i]
            dots.append(examples)
            examples = sigmoid(examples)
            activations.append(examples)

        return examples.T, dots, activations

    # the hard part lmao
    def backward(self, examples, labels):       
        num_examples = examples.shape[1]
        weight_updates = [np.zeros(w.shape) for w in self.weights]
        bias_updates = [np.zeros(b.shape) for b in self.biases]

        output, dots, activations = self.forward(examples)

        for ex in range(num_examples):
            
            # p(Error)/p(Neuron) is initially output-target
            neuron_partials = output[ex]-labels[ex]
            for i in list(range(len(self.weights)))[::-1]:
                # calculate the partials of Error w/ respect to dot product of weights and inputs
                dot_partials = sigmoid_deriv(dots[i][:, ex]) * neuron_partials
                # calculate new neuron partial derivatives w/ respect to the neurons behind
                # this vectorization took me forever to figure out lmao
                neuron_partials = self.weights[i].T.dot(dot_partials)

                # non vectorized
                # for j in range(self.weights[i].shape[1]):
                    # neuron_partials.append(0)
                    # for k in range(len(dot_partials)):
                        # neuron_partials[j] += self.weights[i][k, j] * dot_partials[k]
                
                dot_partials = dot_partials.reshape(len(dot_partials), 1)
                activations_T = activations[i][:,ex].reshape(1, len(activations[i][:,ex]))
                weight_partials = dot_partials.dot(activations_T)
                weight_updates[i] += weight_partials
                bias_updates[i] += dot_partials

        return weight_updates, bias_updates

    def train(self, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE):
        self.display_accuracy()
        for _ in range(epochs):
            # randomize the order
            random_permutation = np.random.permutation(self.train_data.shape[0])
            self.train_data = self.train_data[random_permutation]
            self.train_labels = self.train_labels[random_permutation]

            # perform minibatch SGD
            for start in range(0, int(len(self.train_data) / 20), batch_size):
                print(start)
                end = start + batch_size
                weight_updates, bias_updates = self.backward(self.train_data[start:end].T, self.train_labels[start:end])
                
                for i in range(len(self.weights)):
                    self.weights[i] -= weight_updates[i] * (learning_rate / batch_size)
                    self.biases[i] -= bias_updates[i] * (learning_rate / batch_size)

            self.display_accuracy()

        print('Training finished.')

    def display_accuracy(self):
        error = calc_accuracy(encode_to_onehot(self.forward(self.train_data.T)[0]), self.train_labels)
        print('Training error of {:2}'.format(error)) 
        return error

