from ..base import BaseEstimator

class BaseMLP(BaseEstimator):
    """Base class for estimators base on multi layer
    perceptrons."""

    def __init__(self, n_hidden, lr, l2decay, loss, output_layer, chunk_size):
        self.n_hidden = n_hidden
        self.lr = lr
        self.l2decay = l2decay
        self.loss = loss
        self.batchsize = batchsize

    def fit(self, X, y, max_epochs):
        # get all sizes
        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise ValueError("Shapes of X and y don't fit.")
        n_outs = y.shape[1]
        n_iterations = int(max_epochs * n_batches)
        n_batches = int(np.ceil(float(n_samples) / self.chunk_size))

        # generate batch slices
        batch_slices = list(gen_even_slices(n_samples, n_batches))

        # generate weights.
        # TODO: smart initialization
        self.weights1_ = np.random.uniform(size=(n_features, self.n_hidden))
        self.weights2_ = np.random.uniform(size=(n_features, self.n_hidden))

        # preallocate memory
        x_hidden = np.empty(self.chunk_size, n_features)
        x_output = np.empty(self.chunk_size, n_outs)

        # main loop
        for i, batch_slice in izip(xrange(n_iterations), cycle(batch_slices)):
            self._forward(i, batch_slice, x_hidden, x_output)
            self._backward(i, batch_slice, y_slice)
            pass
        return self

    def predict(self, X):
        return self._forward(X)

    def _forward(self, i, batch_slice):
        pass

    def _backward(self, i, batch_slice, y_slice):
        pass


