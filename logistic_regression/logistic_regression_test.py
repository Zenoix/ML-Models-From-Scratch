from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

from logistic_regression import Logistic_Regression

data = load_breast_cancer(as_frame=True)
log_reg = Logistic_Regression()

X = data["data"]
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

log_reg.fit(X_train, y_train)

y_hat = log_reg.predict(X_test)

print(log_reg.accuracy(y_test, y_hat))
