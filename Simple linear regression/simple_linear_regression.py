import numpy as np


class SimpleLinearRegression:
    def __init__(self, learning_rate=0.001, iterations=2000, verbose=False):
        self.l_rate = learning_rate
        self.iters = iterations
        self.weight = None
        self.bias = None
        self.verbose = verbose

    def __str__(self):
        output = (
            f"Simple Linear Regression Model: "
            f"learning rate={self.l_rate}, iterations={self.iters}"
        )
        return output

    def fit(self, X, y):
        self.weight = 0
        self.bias = 0

        for iteration in range(self.iters):
            y_pred = self.weight*X+self.bias
            dw = np.mean(-2 * X * (y - y_pred))
            db = np.mean(-2 * (y - y_pred))
            self.weight -= self.l_rate * dw
            self.bias -= self.l_rate * db
            if self.verbose and (not iteration % 10 or
                                 iteration == self.iters - 1):
                info = (
                    f"Iteration {iteration}: "
                    f"Weight={self.weight}, Bias={self.bias}"
                )
                print(info)

    def predict(self, X):
        pass

    def params(self):
        return {"Weight": self.weight, "Bias": self.bias}

    def coef(self):
        return self.weight

    def intercept(self):
        return self.bias

    def mse(self, y_true, y_hat):
        return np.mean((y_true - self.weight*y_hat + self.bias) ** 2)


a = SimpleLinearRegression()
