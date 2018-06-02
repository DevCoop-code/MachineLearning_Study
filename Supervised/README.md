# 지도학습
지도학습의 종류
* 분류(Classification)
* 회귀(Regression)

## 분류(Classification)
미리 정의된 클래스 레이블중 하나를 예측하는 것

분류의 종류
* 이진 분류(Binary Classification)
* 다중 분류(Multiclass classification)

### 이진 분류(Binary Classification)
둘 중 하나를 선택하는 것

### 다중 분류(Multiclass Classification)
여러 개 중 하나를 선택하는 것

## 회귀(Regression)
연속적인 데이터를 예측하는 것

# 일반화(Generalization), 과대적합(Overfitting), 과소적합(Underfitting)
## 일반화(Generalization)
모델이 처음 보는 데이터에 대해 정확하게 예측할 수 있는 경우 훈련 세트에서 테스트 세트로 일반화 되었다고 함
모델을 만들때는 가능한 한 정확하게 일반화되도록 해야만 함

## 과대적합(Overfitting)
훈련 데이터로부터 너무 복잡한 모델을 만드는 경우(너무 훈련데이터에 가깝게 맞춰진 모델) 새로운 데이터에 일반화되기 어려울 때 과대적합이 일어났다고 말함

## 과소적합(Underfitting)
모델이 너무 간단한 경우(너무 rough하게 모델을 만드는 경우 이러한 경우 데이터의 다양성을 잡아내지 못하게 됨) 과소적합이 일어났다고 말함