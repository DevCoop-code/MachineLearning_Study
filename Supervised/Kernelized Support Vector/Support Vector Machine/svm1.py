from sklearn.datasets.samples_generator import make_blobs
import mglearn
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

X, y = make_blobs(centers=4, random_state=8)
y = y % 2

linear_svm = LinearSVC().fit(X, y)

mglearn.plots.plot_2d_separator(linear_svm, X)
mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.xlabel("Attribute 0")
plt.ylabel("Attribute 1")
plt.show()