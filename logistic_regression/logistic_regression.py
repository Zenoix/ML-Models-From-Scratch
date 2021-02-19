import numpy as np


class Logistic_Regression:
    def __init__(self, learning_rate=0.01, iterations=1000, verbose=False):
        # initialise hyperparameters
        self.l_rate = learning_rate
        self.iters = iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None

    def __str__(self):
        output = (
            f"Logistic Regression Model: "
            f"learning rate={self.l_rate}, iterations={self.iters}"
        )
        return output
