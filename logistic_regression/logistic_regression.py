# import libraries
import numpy as np
import scipy


class LogisticRegression:
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
            linear = self._linear(X)
            y_pred = self._sigmoid_func(linear)
            loss = y_pred - y

            # gradient descent differentiation
            dw = (1 / n_samples) * np.dot(X.T, loss)
            db = (1 / n_samples) * np.sum(loss)

            # update rules
            self.weights -= self.l_rate * dw
            self.bias -= self.l_rate * db

            if self.verbose and (not iteration % 100 or
                                 iteration == self.iters - 1):
                info = (
                    f"Iteration {iteration}: "
                    f"Weight={self.weights}, Bias={self.bias}"
                )
                print(info)

    def predict(self, X):
        linear = self._linear(X)
        y_pred = self._sigmoid_func(linear)

        # label the binary classes
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        return y_pred

    def params(self):
        return {"Weight": self.weights, "Bias": self.bias}

    def coef(self):
        return self.weights

    def intercept(self):
        return self.bias

    def accuracy(self, y_true, y_hat):
        diff = y_hat - y_true
        return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

    def _linear(self, X):
        return np.dot(X, self.weights) + self.bias

    def _sigmoid_func(self, X):
        # return sigmoid function on each X value
        return scipy.special.expit(X)

    def __str__(self):
        output = (
            f"Logistic Regression Model: "
            f"learning rate={self.l_rate}, iterations={self.iters}"
        )
        return output
