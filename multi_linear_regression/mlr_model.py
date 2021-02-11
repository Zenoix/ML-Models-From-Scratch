import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, verbose=False):
        # initialise hyperparameters
        self.l_rate = learning_rate
        self.iters = iterations
        self.verbose = verbose
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        # initialise parameters
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        # Using gradient descent
        for iteration in range(self.iters):
            # prediction with current parameters
            y_pred = self.weight * X + self.bias

            # gradient descent differentiation
            dw = np.mean(X * (y_pred - y))
            db = np.mean(y_pred - y)

            # update rules
            self.weight -= self.l_rate * dw
            self.bias -= self.l_rate * db

            if self.verbose and (not iteration % 100 or
                                 iteration == self.iters - 1):
                info = (
                    f"Iteration {iteration}: "
                    f"Weight={self.weight}, Bias={self.bias}"
                )
                print(info)

    def predict(self, X):
        return self.weight * X + self.bias

    def params(self):
        return {"Weight": self.weight, "Bias": self.bias}

    def coef(self):
        return self.weight

    def intercept(self):
        return self.bias

    def mse(self, y_true, y_hat):
        return np.mean((y_true - y_hat) ** 2)

    def __str__(self):
        output = (
            f"Simple Linear Regression Model: "
            f"learning rate={self.l_rate}, iterations={self.iters}"
        )
        return output