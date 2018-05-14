from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import mglearn
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from mpl_toolkits.mplot3d import Axes3D, axes3d

X, y = make_blobs(centers=4, random_state=8)
y = y % 2
X_new = np.hstack([X, X[:, 1:]** 2])

linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)

XX, YY = np.meshgrid(xx, yy)
ZZ = YY ** 2

dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()], cmap=mglearn.cm2, alpha=0.5)

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Attribute 0")
plt.ylabel("Attribute 1")
plt.show()