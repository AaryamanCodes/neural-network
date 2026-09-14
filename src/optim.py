class SGD:
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate

    def step(self, params_and_grads):
        for param, grad in params_and_grads:
            param -= self.lr * grad
