# import libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# import the logistic regression model
from logistic_regression import Logistic_Regression

# load data and model
data = load_breast_cancer(as_frame=True)
log_reg = Logistic_Regression()

# obtain features and targets
X = data["data"]
y = data["target"]

# split into testing and training data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

# fit the model to the training data
log_reg.fit(X_train, y_train)

# create predictions on testing features
y_hat = log_reg.predict(X_test)

# print accuracy
print(log_reg.accuracy(y_test, y_hat))
