# import libraries
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# import the model
from simple_linear_regression import SimpleLinearRegression

# simulate linear data
X, y, coef = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, coef=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

print(coef)

# create the model
regressor = SimpleLinearRegression(grad_descent=False)
# train the model
regressor.fit(X_train, y_train)
print(regressor.params())

# make predictions for mse
y_hat = regressor.predict(X_test)

# graph the regression line
pred_line = regressor.predict(X)
plt.scatter(X, y)
plt.plot(X, pred_line, color='black', linewidth=2)
plt.show()

