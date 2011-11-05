from ..base import BaseEstimator

class BaseMLP(BaseEstimator):
    """Base class for estimators base on multi layer
    perceptrons."""

   def __init__(self, num_hidden, lr, l2decay, loss, output_layer, batchsize):
       self.num_hidden = num_hidden
       self.lr = lr
       self.l2decay = l2decay
       self.loss = loss
       self.batchsize = batchsize

   def fit(self, X, y):
       pass
   def predict(self, X):
       pass
