"""
Adaline
=======

Adaptile Linear Neuron

It is a linear classifier that minimizes sum of squared errors SSE

Q(w) = sum((<w,xi> - yi)**2) -> min
                                 w
using gradient descent.

It is basically a simplified regression model, that treats target values {+1, -1} as continuous outcome.

Y = {+1, -1} - target values
Xl = {xi, yi}l - training set
w - weights, w0 - bias unit
discriminative hyperplane f(x,w) = <w,x> = w1*x1 + ... + wn*xn - w0

a(x,w) = sign(<x,w>) - classification algorithm.
"""

import numpy as np


class Adaline:
    """
    Adaptive linear neuron.

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
            errors = y - output
            self.w_[0] += self.eta * errors.sum()
            self.w_[1:] += self.eta * X.T.dot(errors)
            cost = (errors**2).sum() / 2
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input <w,x>=w1*x1+...+wn*xn+w0"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Linear activation"""
        return X

    def predict(self, X):
        """Predict class label(s)"""
        return np.where(self.activation(self.net_input(X)) >= 0, 1, -1)


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('iris.data', header=None)
    # select petal and sepal length
    X = df.iloc[0:100, [0, 2]].values
    # select setosa and versicolor
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    adaline = Adaline(eta=0.0003, n_iter=50, random_state=1)
    adaline.fit(X, y)

    import matplotlib.pyplot as plt
    plt.plot(adaline.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Total cost')
    plt.show()

    from utils import plot_decision_regions
    plot_decision_regions(X, y, adaline)
    plt.xlabel('petal length [cm]')
    plt.ylabel('sepal length [cm]')
    plt.legend(loc='upper left')
    plt.show()
