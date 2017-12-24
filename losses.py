import numpy as np

class Loss():
    def loss(self):
        pass
    def objective(self):
        pass

class ClassificationLoss(Loss):
    def objective(self, examples, labels):
        pass

class SoftmaxCrossEntropyLoss(ClassificationLoss):
    def _softmax(self, x):
        # numerically stable softmax
        x = np.exp(x - np.amax(x, axis=1, keepdims=True))
        sums = np.sum(x, axis=1, keepdims=True)
        return x / sums

    def loss(self, outputs, labels):
        p_locs = np.argmax(labels, axis=1)
        q = self._softmax(outputs)
        # since p only has one non-zero probability,
        # choose those indices from q to make up the loss 
        errors = -np.log(q[range(q.shape[0]), p_locs])
        
        loss = sum(errors) / q.shape[0]
        return loss

    def objective(self, outputs, labels):
        # loss for each examples is - p_c * log(q_c)
        p_locs = np.argmax(labels, axis=1)
        q = self._softmax(outputs)
        
        # gradient of cross entropy softmax is just the softmax outputs
        # subtract p_i. Since p_i is only 1 for 1 entry, subtract accordingly

        q[range(q.shape[0]), p_locs] -= 1
        return q
