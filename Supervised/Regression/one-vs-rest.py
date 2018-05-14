from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=42)
#mglearn.discrete_scatter(X[:, 0],X[:, 1],y)
#plt.xlabel("character 0")
#plt.ylabel("character 1")
#plt.legend(["class 0","class 1","class 2"])
linear_svm = LinearSVC().fit(X, y)
#print("계수 배열의 크기 : ", linear_svm.coef_.shape)
#print("절편 배열의 크기 : ", linear_svm.intercept_.shape)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,mglearn.cm3.colors):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("character 0")
plt.ylabel("character 1")
plt.legend(['class 0','class 1','class 2','class boundary 0','class boundary 1','class boundary 2'], loc=(1.01, 0.3))
plt.show()