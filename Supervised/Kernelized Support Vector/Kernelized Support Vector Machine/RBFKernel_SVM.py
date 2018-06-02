from sklearn.svm import SVC
import numpy as np
import mglearn
import matplotlib.pyplot as plt

X, y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X,y)
mglearn.plots.plot_2d_separator(svm, X, eps=.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

#Support Vector
sv = svm.support_vectors_

#Decide the label of support vector depend on dual_coef_
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:,0], sv[:,1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel("Attribute 0")
plt.xlabel("Attribute 1")
plt.show()