import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from numpy import linalg as LA

def square(x, y):
    return x * x + y * y


# 2d plot
points = np.random.rand(10, 2) * 10
plt.figure()
plt.plot(points[:, 0], points[:, 1], '.y')

covm = np.cov(points.T)
print covm

pca = PCA()
pca.fit(points)
print pca.components_

p1 = np.zeros(points.shape[0])
p2 = np.zeros(points.shape[0])
for i in range(0, points.shape[0]):
    p1[i] = np.dot(pca.components_[0, :], points[i, :])
    p2[i] = np.dot(pca.components_[1, :], points[i, :])

translate1 = 0.5 *((p1.max() * pca.components_[0, :])/square(pca.components_[0, 0], pca.components_[0, 1])
                    + (p1.min() * pca.components_[0, :])/square(pca.components_[0, 0], pca.components_[0, 1]))

translate2 = 0.5 *((p2.max() * pca.components_[1, :])/square(pca.components_[1, 0], pca.components_[1, 1])
                    + (p2.min() * pca.components_[1, :])/square(pca.components_[1, 0], pca.components_[1, 1]))

center = translate1 + translate2
plt.plot(center[0], center[1], '.r')

l1 = 0.5 * (p1.max() - p1.min()) / math.sqrt(square(pca.components_[0, 0], pca.components_[0, 1]))
l2 = 0.5 * (p2.max() - p2.min()) / math.sqrt(square(pca.components_[1, 0], pca.components_[1, 1]))
d1 = l1 * pca.components_[0, :] + l2 * pca.components_[1, :]
d2 = (-1) * l1 * pca.components_[0, :] + l2 * pca.components_[1, :]
d3 = l1 * pca.components_[0, :] + (-1) * l2 * pca.components_[1, :]
d4 = (-1) * l1 * pca.components_[0, :] + (-1) * l2 * pca.components_[1, :]
v1 = np.zeros([5, 2])
v1[0, :] = center + d1
v1[1, :] = center + d3
v1[2, :] = center + d4
v1[3, :] = center + d2
v1[4, :] = center + d1
plt.plot(v1[:, 0], v1[:, 1], center[1], 'r')

plt.xlim(0, 20)
plt.xticks(range(0, 20, 1))
plt.ylim(0, 20)
plt.show()


points = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [3, 3, 4],
                   [3, 3, 2], [2, 3, 3], [3, 2, 3], [4, 3, 3], [3, 4, 3]])

covm = np.cov(points.T)
print covm
w, v = LA.eig(covm)
print w
print v
print np.cross(v[:, 0], v[:, 1])

pca = PCA()
pca.fit(points)
print pca.components_

