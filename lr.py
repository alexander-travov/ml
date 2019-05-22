"""
Logistic regression
===================

Logistic regression solves the task of binary classification.

You can come to the LR model from two sides:
- Linear discriminative model that minimizes empiric risk
  with logarithmic loss function L(M)=ln(1+exp(-M)) or
- Optimal Bayes classification that maximizes aposteriori likelihood
  under the assumtion of exponent conditional densities in classes.

Those two are equivalent.

Y = {+1, -1} - target values
Xl = {xi, yi}l - training set
w - weights, w0 - bias unit
discriminative hyperplane f(x,w) = <w,x> = w1*x1 + ... + wn*xn - w0

a(x,w) = sign(<x,w>) - classification algorithm.

Aposteriori class probabilities for x are given by:
P(y|x) = sigma(f(x,w)y), where sigma(z) = 1/(1+exp(-z)) - logistic sigmoid
"""

import numpy as np


class LR:
    """
    Logistic regression using batch gradient descent.

    Parameters:
    -----------

    eta: float
       Learning rate (between 0. and 1.).

    n_iter: int
       Number of passes over the training dataset.

    random_state: int
       Random number generator seed for weight initialization.

    Attributes:
    -----------

    w_: 1d-array
       Weights after fitting.

    cost_: list
       Total cost (SSE) in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit training data.

        Parameters:
        -----------

        X: array-like, shape = [n_samples, n_features]
           Training vectors

        y: array-like, shape = [n_samples]
           Target values

        Returns:
        --------

        self
        """
        # Initializing weights to small gaussian variables
        rgen = np.random.RandomState(seed=self.random_state)
        self.w_ = rgen.normal(loc=0, scale=0.01, size=1+X.shape[1])

        self.cost_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = 1 + y - 2*output # for classes are {-1;+1}
            # errors = y - output # for classes are {0;1}
            self.w_[0] += self.eta * errors.sum()
            self.w_[1:] += self.eta * X.T.dot(errors)
            cost = (errors**2).sum() / 2
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input <w,x>=w1*x1+...+wn*xn+w0"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Logistic sigmoid activation"""
        return 1./(1. + np.exp(-np.clip(X, -250, 250)))

    def predict(self, X):
        """Predict class label(s)"""
        return np.where(self.net_input(X) >= 0, 1, -1)
        # return np.where(self.net_input(X) >= 0, 1, 0)


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('iris.data', header=None)
    # select petal and sepal length
    X = df.iloc[0:100, [0, 1]].values

    # Feature standardization
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean())/X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean())/X[:,1].std()

    # select setosa and versicolor
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    # y = np.where(y == 'Iris-setosa', 0, 1)

    lr = LR(eta=0.05, n_iter=150, random_state=1)
    lr.fit(X_std, y)

    import matplotlib.pyplot as plt
    plt.plot(lr.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Total cost')
    plt.show()

    from utils import plot_decision_regions
    plot_decision_regions(X_std, y, lr)
    plt.xlabel('petal length [cm]')
    plt.ylabel('sepal length [cm]')
    plt.legend(loc='upper left')
    plt.show()
