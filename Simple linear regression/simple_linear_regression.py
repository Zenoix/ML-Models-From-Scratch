import numpy as np


class SimpleLinearRegression:
    def __init__(self, learning_rate=0.001, iterations=2000, verbose=False):
        self.l_rate = learning_rate
        self.iters = iterations
        self.weight = None
        self.bias = None

    def __str__(self):
        output = f"SimpleLinearRegression: learning rate={self.l_rate}, iterations={self.iters}"
        return output

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def params(self):
        return {"Weight": self.weight, "Bias": self.bias}

    def coef(self):
        return self.weight

    def intercept(self):
        return self.bias

    def mse(self):
        pass
