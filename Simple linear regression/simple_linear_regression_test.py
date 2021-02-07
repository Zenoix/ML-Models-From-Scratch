# import libraries
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# import the model
from simple_linear_regression import SimpleLinearRegression

# simulate linear data
X, y = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# create the model
regressor = SimpleLinearRegression()

# train the model
regressor.fit(X_train, y_train)

# model parameters
print(regressor.params())

# make predictions for mse
y_hat = regressor.predict(X_test)
print(regressor.mse(y_test, y_hat))

# graph the regression line
pred_line = regressor.predict(X)
plt.scatter(X, y)
plt.plot(X, pred_line, color='black', linewidth=1)
plt.show()
