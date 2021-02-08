# Simple Linear Regression

## Formula

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;y=wx&plus;b" title="y=wx+b" />

- <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;y" title="y" /> is the target
- <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;w" title="w" /> is the weight/slope
- <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;x" title="x" /> is the feature
- <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;b" title="b" /> is the bias/intercept

## Cost Function

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;MSE&space;=&space;J(w,b)&space;=&space;\frac{1}{N}\sum_{i=1}^n(y_{i}-(wx_{i}&space;&plus;&space;b))^2" title="MSE = J(w,b) = \frac{1}{N}\sum_{i=1}^n(y_{i}-(wx_{i} + b))^2" />

- <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;MSE" title="MSE" /> or mean squared error is the average of the squared values of the difference between the true target and the predicted target
- <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\frac{1}{N}\sum_{i=1}^{n}" title="\frac{1}{N}\sum_{i=1}^{n}" />is the mean of errors
- <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;y_{i}" title="y_{i}" /> is the true value of an observation and <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;wx_{i}+b" title="wx_{i}+b" /> is our prediction

## Gradient Descent Equation

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;{J}'(w,b)&space;=&space;\begin{bmatrix}&space;\frac{dJ}{dw}&space;\\&space;\frac{dJ}{db}&space;\end{bmatrix}&space;=&space;\begin{bmatrix}&space;\frac{1}{N}\sum_{i=1}^n-2x_{i}(y_{i}-(wx_{i}&space;&plus;&space;b))&space;\\&space;\frac{1}{N}\sum_{i=1}^n-2(y_{i}-(wx_{i}&space;&plus;&space;b))&space;\end{bmatrix}" title="J^{'}(w,b) = \begin{bmatrix} \frac{dj}{dw} \\ \frac{dj}{db} \end{bmatrix} = \begin{bmatrix} \frac{1}{N}\sum_{i=1}^n-2x_{i}(y_{i}-(wx_{i} + b)) \\ \frac{1}{N}\sum_{i=1}^n-2(y_{i}-(wx_{i} + b)) \end{bmatrix}" />

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;{J}'(w,b)" title="{J}'(w,b)" /> and <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\begin{bmatrix}&space;\frac{dJ}{dw}\\&space;\frac{dJ}{db}&space;\end{bmatrix}" title="\begin{bmatrix} \frac{dJ}{dw}\\ \frac{dJ}{db} \end{bmatrix}" /> being the derivatives of the the cost function

##  Learning rules

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;w_{new}&space;=&space;w_{old}&space;-&space;\alpha&space;\cdot&space;dw" title="w_{new} = w_{old} - a \cdot dw" />

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;b_{new}&space;=&space;b_{old}&space;-&space;\alpha&space;\cdot&space;db" title="b_{new} = b_{old} - a \cdot db" />

- <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;w_{new}" title="w" /> is the new weights that come from updating <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;w_{old}" title="w" />

- <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;dw" title="dw" /> is the derivative of the weights from the gradient descent function

- <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;b_{new}" title="b" /> is the new weights that come from updating <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;b_{old}" title="b" />

- <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;db" title="db" /> is the derivative of the bias from the gradient descent function

- <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\alpha" title="\alpha" /> is the previously set learning rate of the linear regression model

  