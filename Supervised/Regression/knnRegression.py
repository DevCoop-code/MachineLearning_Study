from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_wave(n_samples = 40)

#훈련 세트와 테스트 세트로 나눔
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

#이웃의 수를 3으로 하여 모델의 객체를 만듬
reg = KNeighborsRegressor(n_neighbors = 3)

#훈련 데이터와 타깃을 사용하여 모델을 학습
reg.fit(X_train, y_train)

#테스트 세트에 대해 예측
#print("테스트 세트 예측:\n{}".format(reg.predict(X_test)))

print("테스트 세트 : {:.2f}".format(reg.score(X_test, y_test)))

plt.show()