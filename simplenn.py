import numpy as np
import scipy.io

class SimpleNN():
    INPUT_SIZE = 784
    # Each entry represents a layer of size n
    HIDDEN_LAYERS = [200, 200]
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
            weight_matrix = np.random.rand(all_layer_sizes[i], all_layer_sizes[i-1]) - 0.5
            bias_matrix = np.random.rand(all_layer_sizes[i])
            self.weights.append(weight_matrix)
            self.biases.append(bias_matrix)

        train_dataset = scipy.io.loadmat(self.TRAIN_DATA_LOC)
        test_dataset = scipy.io.loadmat(self.TEST_DATA_LOC)
        self.train_data = train_dataset['train_images']
        self.train_labels = train_dataset['train_labels']
        self.test_data = test_dataset['test_images']
        self.test_labels = test_dataset['test_images']

    def forward(self):
        return
    def backward(self):
        return
    def train(self):
        return
    def test(self):
        return

