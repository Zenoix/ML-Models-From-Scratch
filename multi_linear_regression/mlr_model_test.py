# import libraries
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import random

# import the model
from slr_model import SimpleLinearRegression

# generate linear data
X, y, coef = datasets.make_regression(
    n_samples=1000, n_features=1, noise=20, coef=True)

# need to randomly make negative data sometimes
negative = random.choice([1, -1])
y, coef = y * negative, coef * negative

# data's coefficient
print(coef)

# split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

# create the model
regressor = SimpleLinearRegression()

# train the model
regressor.fit(X_train, y_train)

# model parameters
print(regressor.params())

# make predictions for mse
y_hat = regressor.predict(X_test)

# print mse
print(regressor.mse(y_test, y_hat))

# graph the regression line
pred_line = regressor.predict(X)
plt.scatter(X, y)
plt.plot(X, pred_line, color='black', linewidth=2)
plt.show()
