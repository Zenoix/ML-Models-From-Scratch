# Simple Linear Regression

## Resources used
- https://en.wikipedia.org/wiki/Simple_linear_regression (Concept explanation and some of the maths behind it)
- https://medium.com/towards-artificial-intelligence/linear-regression-complete-derivation-406f2859a09a (Math behind the non gradient descent method)
- https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html# (Concept explanations and gradient descent maths)
- https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html (Gradient descent explanation and maths)

### Bad animation lol
![Model in action](https://cdn.discordapp.com/attachments/644342031279194112/808173400161910794/fsd.gif)

## How to use the model

#### Import the class
```py
from slr_model import SimpleLinearRegression
```
#### Create the model class
```py
lr = SimpleLinearRegression()
```
##### Possible arguments:
- `learning_rate` which controls how much the weight and bias changes (default set to 0.01)
- `iterations` controls how many iterations the model trains (default set to 1000)
- `grad_descent` is a boolean value where True means the model will use gradient descent to train, False means it will use a non gradient descent method (True by default)
- `verbose` is another boolean value. If set to True, the model will print the iteration number and the current parameters every 100 iterations (False by default)

#### Train the model
```py
lr.fit(X_train, y_train)
```
Both parameters must be numpy arrays (pandas series have not been tested yet). Returns nothing.

#### Make predictions
```py
lr.predict(X)
```
X must be a numpy array (pandas series have not been tested yet). Returns a numpy array of predicted values.

---
### Extra methods

#### Get the parameters of the model
```py
lr.params()
```
Returns a dictionary of the current weight and bias of the model.

#### Get the coefficient of the data
```py
lr.coef()
```
Returns the coefficient of the data/weight of the model.

#### Get the intercept of the data
```py
lr.intercept()
```
Returns the y-intercept of the data/bias of the model.

#### Get the mean squared error
```py
lr.mse()
```
Returns the mean squared error of the model.

#### Print out the hyperparamters of the model
```py
print(lr)
```
Prints out the learning rate and number of iterations.
