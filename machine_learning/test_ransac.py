#! coding:utf-8

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt

X, y = make_regression(n_samples=100, n_features=1, noise=5.0, random_state=0)
test_X = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]).reshape(-1, 1)

# reg = RANSACRegressor(random_state=0).fit(X, y)
reg = LinearRegression().fit(X, y)
print(reg.score(X, y))
test_y = reg.predict(test_X)

plt.figure('test_ransac', figsize=(12, 8))
plt.scatter(X, y, c='black')
plt.scatter(test_X, test_y, c='red')
plt.show()
