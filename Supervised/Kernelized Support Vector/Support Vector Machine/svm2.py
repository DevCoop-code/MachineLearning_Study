from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import mglearn
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from mpl_toolkits.mplot3d import Axes3D, axes3d

X, y = make_blobs(centers=4, random_state=8)
X_new = np.hstack([X, X[:, 1:]** 2])

figure = plt.figure()

#3-dimension graph
ax = Axes3D(figure, elev=-152, azim=-26)

# Draw the y == 0 point and then draw y == 1 point
mask = y == 0
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60, edgecolor='k')
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60, edgecolor='k')
ax.set_xlabel("Attribute 0")
ax.set_ylabel("Attribute 1")
ax.set_zlabel("Attribute 1 ** 2")
plt.show()