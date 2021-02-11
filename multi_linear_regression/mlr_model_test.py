# import libraries
from sklearn.model_selection import train_test_split
from sklearn import datasets
import random

# import the model
from mlr_model import LinearRegression

# generate linear data
X, y, coef = datasets.make_regression(
    n_samples=10, n_features=random.randint(2, 15), noise=20, coef=True)

# split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

# create the model
regressor = LinearRegression()

# train the model
regressor.fit(X_train, y_train)

# model parameters
print(regressor.params())

# make predictions for mse
y_hat = regressor.predict(X_test)

# print mse
print(regressor.mse(y_test, y_hat))
