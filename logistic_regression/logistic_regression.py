import numpy as np


class Logistic_Regression:
    def __init__(self, learning_rate=0.01, iterations=1000, verbose=False):
        # initialise hyperparameters
        self.l_rate = learning_rate
        self.iters = iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # initialise parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Using gradient descent
        for iteration in range(self.iters):
            # prediction with current parameters
            y_pred = self._linear(X)

            # TODO check if this derivative is correct for logistic reg
            derivative = np.dot(X.T, y_pred - y)

            # update rules
            self.weights -= self.l_rate * derivative
            self.bias -= self.l_rate * derivative

    def predict(self, X):
        y_pred = self._linear(X)
        y_pred = self._sigmoid_func(y_pred)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        return y_pred

    def score(y_pred, y_true):
        diff = y_pred - y_true
        return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

    def _linear(self, X):
        return np.dot(X.T, self.weights) + self.bias

    def _sigmoid_func(self, X):
        return 1 / (1 + np.exp(-X))

    def __str__(self):
        output = (
            f"Logistic Regression Model: "
            f"learning rate={self.l_rate}, iterations={self.iters}"
        )
        return output
