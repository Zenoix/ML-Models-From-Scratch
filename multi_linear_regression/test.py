# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sklearn.model_selection import train_test_split
from sklearn import datasets



# %%
# import the model
from mlr_model import LinearRegression


# %%
X, y, coef = datasets.make_regression(
    n_samples=10, n_features=2, noise=20, coef=True)


# %%
# split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)


# %%
lr = LinearRegression()
lr.fit(X_train, y_train)


# %%

print(lr.weights)
print(coef)
