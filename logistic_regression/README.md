# Logistic Regression

## Resources Used
- https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html# (Concepts and maths)
- https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148 (Maths)
- https://en.wikipedia.org/wiki/Logistic_regression

## How to use the model
#### Import the class
```py
from logistic_regression import LogisticRegression
```
#### Create the model class
```py
log_reg = LogisticRegression()
```
##### Possible arguments:
- `learning_rate` which controls how much the weight and bias changes (default set to 0.01)
- `iterations` controls how many iterations the model trains (default set to 1000)
- `verbose` is another boolean value. If set to True, the model will print the iteration number and the current parameters every 100 iterations (False by default)

#### Train the model
```py
log_reg.fit(X_train, y_train)
```
Trains the model and adjusts the weights and bias to fit the data given. Returns nothing

#### Make predictions
```py
y_hat = log_reg.predict(X)
```
Returns a numpy array of predicted values.

---
### Extra methods

#### Get the parameters of the model
```py
log_reg.params()
```
Returns a dictionary of the current weights and bias of the model before the logistic activation.

#### Get the coefficient of the data
```py
log_reg.coef()
```
Returns the coefficients of the data/weights of the model before the logistic activation.

#### Get the intercept of the data
```py
log_reg.intercept()
```
Returns the y-intercept of the data/bias of the model before the logistic activation.

#### Get the mean squared error
```py
log_reg.accracy()
```
Returns the percentage of values correctly predicted.

#### Print out the hyperparamters of the model
```py
print(log_reg)
```
Prints out the learning rate and number of iterations.