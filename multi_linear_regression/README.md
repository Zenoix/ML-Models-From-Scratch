# Multivariate Linear Regression
## Resources used

## How to use the model

#### Import the class
```py
from mlr_model import LinearRegression
```
#### Create the model class
```py
lr = LinearRegression()
```
##### Possible arguments:
- `learning_rate` which controls how much the weight and bias changes (default set to 0.01)
- `iterations` controls how many iterations the model trains (default set to 1000)
- `verbose` is another boolean value. If set to True, the model will print the iteration number and the current parameters every 100 iterations (False by default)

#### Train the model
```py
lr.fit(X_train, y_train)
```
Trains the model and adjusts the weights and bias to fit the data given. Returns nothing

#### Make predictions
```py
lr.predict(X)
```
Returns a numpy array of predicted values.

---
### Extra methods

#### Get the parameters of the model
```py
lr.params()
```
Returns a dictionary of the current weights and bias of the model.

#### Get the coefficient of the data
```py
lr.coef()
```
Returns the coefficients of the data/weights of the model.

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
