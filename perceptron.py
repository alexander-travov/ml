"""
Perceptron classifier
=====================

Linear classifier that implements Rosenblatt's learning rule, that is derived
using stochastic gradient descent over Hebb's loss function:

Y = {+1, -1} - target values
Xl = {xi, yi}l - training set
w - weights, w0 - bias unit
discriminative hyperplane f(x,w) = <w,x> = w1*x1 + ... + wn*xn - w0
M = <w,x>y - margin of the element. Negative on misclassifications.

a(x,w) = sign(<x,w>) - classification algorithm.

Empiric risk: sum(L(Mi))l where loss function:

L(M) = -M if M<0 else 0

Derivative of L:
L'(M) = -1 if M<0 else 0, so weight updates will happen only on misclassifications.
"""

import numpy as np


class Perceptron:
    """
    Perceptron classifier.

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

    errors_: list
       Number of misclassifications (updates) in each epoch.
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

        self.errors_ = []

        for _ in range(self.n_iter):
            n_errors = 0
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                # updates are made only on misclassifications
                if prediction == target:
                    continue
                update = - self.eta * prediction
                self.w_[0] += update
                self.w_[1:] += update * xi
                n_errors += 1
            self.errors_.append(n_errors)
        return self

    def net_input(self, X):
        """Calculate net input <w,x>=w1*x1+...+wn*xn+w0"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Predict class label(s)"""
        return np.where(self.net_input(X) >= 0, 1, -1)


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('iris.data', header=None)
    # select petal and sepal length
    X = df.iloc[0:100, [0, 2]].values
    # select setosa and versicolor
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    perceptron = Perceptron(eta=0.01, n_iter=10, random_state=1)
    perceptron.fit(X, y)

    import matplotlib.pyplot as plt
    plt.plot(perceptron.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.show()

    from utils import plot_decision_regions
    plot_decision_regions(X, y, perceptron)
    plt.xlabel('petal length [cm]')
    plt.ylabel('sepal length [cm]')
    plt.legend(loc='upper left')
    plt.show()
