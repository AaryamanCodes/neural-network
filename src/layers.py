import numpy as np

class Linear:

    def __init__(self, in_features, out_features, rng=None):

        rng = rng or np.random.default_rng()
        self.W = rng.standard_normal((in_features, out_features)) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):

        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        dx = dout @ self.W.T
        return dx