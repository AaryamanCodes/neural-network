import numpy as np
from src.layers import Linear, ReLU, SoftmaxCrossEntropy


class NeuralNetwork:

    def __init__(self, in_features, hidden_features, out_features, seed=0):
        rng = np.random.default_rng(seed)
        self.fc1 = Linear(in_features, hidden_features, rng=rng)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_features, out_features, rng=rng)
        self.loss_layer = SoftmaxCrossEntropy()

    def forward(self, x, y):
    
            z1 = self.fc1.forward(x)
            a1 = self.relu.forward(z1)
            logits = self.fc2.forward(a1)
            loss, probs = self.loss_layer.forward(logits, y)
            return loss, probs
    
    def backward(self):
    
        dlogits = self.loss_layer.backward()
        da1 = self.fc2.backward(dlogits)
        dz1 = self.relu.backward(da1)
        _dx = self.fc1.backward(dz1)
        return _dx
    