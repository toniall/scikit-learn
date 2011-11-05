import numpy as np
import warnings

from itertools import cycle, izip

from ..utils import gen_even_slices
from ..base import BaseEstimator
from ..utils import shuffle

def softmax_(x):
    #import ipdb
    #ipdb.set_trace()
    np.exp(x,x)
    x /= np.sum(x, axis=1)[:, np.newaxis]

class BaseMLP(BaseEstimator):
    """Base class for estimators base on multi layer
    perceptrons."""

    def __init__(self, n_hidden, lr, l2decay, loss, output_layer, chunk_size):
        self.n_hidden = n_hidden
        self.lr = lr
        self.l2decay = l2decay
        self.loss = loss
        self.chunk_size = chunk_size

    def fit(self, X, y, max_epochs, shuffle_data):
        # get all sizes
        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise ValueError("Shapes of X and y don't fit.")
        self.n_outs = y.shape[1]
        #n_batches = int(np.ceil(float(n_samples) / self.chunk_size))
        n_batches = n_samples / self.chunk_size
        if n_samples % self.chunk_size != 0:
            warnings.warn("Discarding some samples: \
                sample size not divisible by chunk size.")
        n_iterations = int(max_epochs * n_batches)

        if shuffle_data:
            X_shuffled = shuffle(X)
        # generate batch slices
        batch_slices = list(gen_even_slices(n_batches * self.chunk_size, n_batches))

        # generate weights.
        # TODO: smart initialization
        self.weights1_ = np.random.uniform(size=(n_features, self.n_hidden))/np.sqrt(n_features)
        self.bias1_ = np.zeros(self.n_hidden)
        self.weights2_ = np.random.uniform(size=(self.n_hidden, self.n_outs))/np.sqrt(self.n_hidden)
        self.bias2_ = np.zeros(self.n_outs)

        # preallocate memory
        x_hidden = np.empty((self.chunk_size, self.n_hidden))
        delta_h = np.empty((self.chunk_size, self.n_hidden))
        x_output = np.empty((self.chunk_size, self.n_outs))
        delta_o = np.empty((self.chunk_size, self.n_outs))

        # main loop
        for i, batch_slice in izip(xrange(n_iterations), cycle(batch_slices)):
            self._forward(i, X_shuffled, batch_slice, x_hidden, x_output)
            self._backward(i, X_shuffled, y, batch_slice, x_hidden, x_output, delta_o, delta_h)
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        x_hidden = np.empty((n_samples, self.n_hidden))
        x_output = np.empty((n_samples, self.n_outs))
        self._forward(None, X, slice(0, n_samples), x_hidden, x_output)
        return x_output

    def _forward(self, i, X, batch_slice, x_hidden, x_output):
        """Do a forward pass through the network"""
        x_hidden[:] = np.dot(X[batch_slice], self.weights1_)
        x_hidden += self.bias1_
        np.tanh(x_hidden, x_hidden)
        x_output[:] = np.dot(x_hidden, self.weights2_)
        x_output += self.bias2_
        softmax_(x_output)
        #np.tanh(x_output, x_output)

    def _backward(self, i, X, y, batch_slice, x_hidden, x_output, delta_o, delta_h):
        """Do a backward pass through the network and update the weights"""
        delta_o[:] = x_output - y[batch_slice] # TODO: copy?
        #import ipdb
        #ipdb.set_trace()
        print(np.linalg.norm(delta_o/self.chunk_size))
        #delta_o *= 1. - x_output**2
        delta_o *= x_output
        delta_h[:] = np.dot(delta_o, self.weights2_.T)

        # update weights
        self.weights2_ -= self.lr/self.chunk_size * np.dot(x_hidden.T, delta_o)
        self.bias2_ -= self.lr * np.mean(delta_o, axis=0)
        self.weights1_ -= self.lr/self.chunk_size * np.dot(X[batch_slice].T, delta_h)
        self.bias1_ -= self.lr * np.mean(delta_h, axis=0)


