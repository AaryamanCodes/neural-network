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

    def params_and_grads(self):
            return [(self.W, self.dW), (self.b, self.db)]

class ReLU:

    def __init__(self):
        self.mask = None
    def forward(self, z):
        self.mask = (z > 0)
        return z * self.mask
    def backward(self, dout):
        return dout * self.mask


class SoftmaxCrossEntropy:

    def __init__(self):
        self.probs = None
        self.y = None
        self.n = None

    def forward(self, logits, y):
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        n = logits.shape[0]
        log_likelihood = -np.log(probs[np.arange(n), y] + 1e-12)
        loss = log_likelihood.mean()

        self.probs, self.y, self.n = probs, y, n
        return loss, probs

    def backward(self):
        dlogits = self.probs.copy()
        dlogits[np.arange(self.n), self.y] -= 1
        dlogits /= self.n
        return dlogits