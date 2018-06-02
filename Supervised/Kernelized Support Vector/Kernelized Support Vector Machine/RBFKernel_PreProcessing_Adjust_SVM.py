from sklearn.svm import SVC
import numpy as np
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

#Minimum data Each Attributes
min_on_training = X_train.min(axis=0)
min_on_testing = X_test.min(axis=0)
#Calculate the range(Maximum value - Minimum value)
range_on_training = (X_train - min_on_training).max(axis=0)
#Adjust PreProcessing X-min(x)/max(X)-min(X)
X_train_scaled = (X_train - min_on_training) / range_on_training
X_test_scaled = (X_test - min_on_testing) / range_on_training

svc = SVC(C=1000)
svc.fit(X_train_scaled, y_train)

print("Accuracy about Training Set : {:.2f}".format(svc.score(X_train_scaled, y_train)))
print("Accuracy about Test Set : {:.2f}".format(svc.score(X_test_scaled, y_test)))