import numpy as np
import scipy.io
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

sigmoid = np.vectorize(sigmoid)

def encode_to_onehot(output):
    output = np.eye(output.shape[1])[list(np.argmax(output, 1))]
    return output

def calc_accuracy(output, truth):
    assert len(output) == len(truth), 'Output and truth must be same length'
    return np.sum(output == truth) / len(output)

class SimpleNN():
    # HYPER PARAMETERS
    # Each entry represents a layer of size n
    # These are just defaults, can be set train() function as well
    HIDDEN_LAYERS = [200, 200]
    NUM_EPOCHS = 50
    BATCH_SIZE = 100
    LEARNING_RATE = 0.01

    # Each entry represents a layer of size n
    HIDDEN_LAYERS = [200, 200]
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
        self.test_labels = test_dataset['test_images']

        # simple normalization
        self.train_data = (self.train_data / np.amax(self.train_data))
        self.test_data = (self.test_data / np.amax(self.test_data))
        
        # reshape data into matrix of (m, feature_size)
        self.train_data = np.swapaxes(self.train_data.reshape((self.INPUT_SIZE, self.train_data.shape[2])), 0, 1)
        self.test_data = np.swapaxes(self.test_data.reshape((self.INPUT_SIZE, self.test_data.shape[2])), 0, 1)

    def forward(self, examples):
        assert examples.shape[0] == self.INPUT_SIZE, 'Input vectors must be of size ' + self.INPUT_SIZE

        for i in range(len(self.weights)):
            examples = self.weights[i].dot(examples)
            examples += self.biases[i]
            examples = sigmoid(examples)
            
        return examples

    # the hard part lmao
    def backward(self, examples, labels):       
        return examples

    def train(self, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE):
        for _ in range(epochs):
            # randomize the order
            random_permutation = np.random.permutation(batch_size)
            example_data = self.train_data[random_permutation]
            example_labels = self.train_labels[random_permutation]

            # perform minibatch SGD
            for start in range(0, len(self.train_data), batch_size):
                end = start + batch_size
                a = self.backward(example_data[start:end], example_labels[start:end])
                print(a)

    def test(self):
        error = calc_accuracy(encode_to_onehot(self.forward(self.test_data)), self.test_labels)
        print('Test error of {0}'.format(error)) 
        return error

