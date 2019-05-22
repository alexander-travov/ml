"""
This example shows the influence of L2 regilarization parameter of weights norm
for logistic regression.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.3,
    random_state=1,
    stratify=y
)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

weights, params = [], []
for c in range(-5, 5):
    C = 10.**c
    params.append(C)
    lr = LogisticRegression(C=C, solver='lbfgs', multi_class='ovr', random_state=1)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
weights = np.array(weights)

plt.plot(params, weights[:,0], label='W petal length')
plt.plot(params, weights[:,1], label='W petal width', linestyle='--')
plt.xlabel('C')
plt.ylabel('weights')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()
