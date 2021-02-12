import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.001, iterations=1000, verbose=False):
        # initialise hyperparameters
        self.l_rate = learning_rate
        self.iters = iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None

    def _normalize_features(self, X):
        # mean normalization
        for feature in X.T:
            fmean = np.mean(feature)
            frange = np.amax(feature) - np.amin(feature)
            feature -= fmean
            feature /= frange
        return X

    def fit(self, X, y):
        # initialise parameters
        n_samples, n_features = X.shape
        #X = self._normalize_features(X)
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Using gradient descent
        for iteration in range(self.iters):
            # prediction with current parameters
            y_pred = self.predict(X)
            error = y_pred - y

            # gradient descent differentiation
            dw = np.dot(X.T, error) / n_samples
            db = np.mean(error)

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
        #X = self._normalize_features(X)
        return np.dot(X, self.weights) + self.bias

    def params(self):
        return {"Weight": self.weights, "Bias": self.bias}

    def coef(self):
        return self.weights

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
