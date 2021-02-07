import numpy as np


class SimpleLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000,
                 grad_descent=True, verbose=False):
        self.l_rate = learning_rate
        self.iters = iterations
        self.weight = None
        self.bias = None
        self.grad_descent = grad_descent
        self.verbose = verbose

    def fit(self, X, y):
        self.weight = 0
        self.bias = 0
        N = len(X)
        if self.grad_descent:
            for iteration in range(self.iters):
                y_pred = self.weight * X + self.bias
                dw = (1 / N) * np.sum(X * (y_pred - y))
                db = (1 / N) * np.sum(y_pred - y)

                self.weight -= self.l_rate * dw
                self.bias -= self.l_rate * db

                if self.verbose and (not iteration % 100 or
                                     iteration == self.iters - 1):
                    info = (
                        f"Iteration {iteration}: "
                        f"Weight={self.weight}, Bias={self.bias}"
                    )
                    print(info)
        else:
            X = X.T[0]
            y_mean = y.mean()
            X_mean = X.mean()
            sum_Xy = (y * X).sum()
            Xy_byN = (y.sum() * X.sum()) / N
            sum_X_square = (X * X).sum()
            X_square_byN = (X.sum() * X.sum()) / N

            self.weight = (sum_Xy - Xy_byN) / (sum_X_square - X_square_byN)
            self.bias = y_mean - self.weight * X_mean

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
