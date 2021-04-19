"""
피처 스케일링은 서로 다른 변수의 값 범위를 일정한 수준으로 맞추는 작업이다.
피처 스케일링에는 대표적으로 두 가지 방법이 있다.

1. Standardization (StandardScalar)
표준화는 가우시안 정규 분포를 따르도록 데이터의 피처를 변환하는 작업이다.
여기서 가우시안 정규 분포는 평균이 0, 분산이 1인 분포를 의미한다.
서포트 벡터 머신, 선형회귀, 로지스틱 회귀에서는 데이터가 가우시안 정규 분포를 따르는 것을 가정한다.

표준화는 보통 하나의 데이터 그룹에 대해 표준화를 할 때 사용한다.
특정 데이터가 데이터 그룹에서 어느 위치에 있는지 파악하기 위함.

xi_new = (xi - mean(x)) / stdev(x)
mean : 평균
stdev : 표준편차

2. Normalization (MinMaxScalar)
정규화는 서로 다른 데이터 그룹의 단위를 표준화하기 위한 작업이다.
예를 들어, 특정 데이터들의 키라는 피처 데이터 그룹과 몸무게라는 피처 데이터그룹이 있을 때
서로 다른 단위를 가진 키, 몸무게 피처 데이터 그룹을 0~1로 정규화하는 것을 의미한다.

키가 최소 160~180cm, 몸무게 50~80kg
>> 키 : 0~1, 몸무게 0~1

xi_new = (xi-min(x)) / (max(x)-min(x))
"""
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris


# Standard Scaler
iris = load_iris()
iris_feature_names = iris.feature_names
iris_data = iris.data
iris_df = pd.DataFrame(data=iris_data, columns=iris_feature_names)

print(iris_df, '\n')
print(iris_df.mean(), '\n')  # feature 들의 평균값
print(iris_df.var(), '\n')  # feature 들의 분산값

standard_scaler = StandardScaler()
standard_scaler.fit(iris_df)
iris_scaled = standard_scaler.transform(iris_df)
iris_scaled_df = pd.DataFrame(data=iris_scaled, columns=iris_feature_names)

print(iris_scaled, '\n')
print(iris_scaled_df, '\n')
print(iris_scaled_df.mean(), '\n')  # 스케일링된 feature 들의 평균값
print(iris_scaled_df.var(), '\n')  # 스케일링된 feature 들의 분산값


# MinMax Scaler
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(iris_df)
iris_normalized = min_max_scaler.transform(iris_df)
iris_normalized_df = pd.DataFrame(data=iris_normalized, columns=iris_feature_names)

print(iris_normalized_df, '\n')
print(iris_normalized_df.min(), '\n')  # 정규화된 feature 들의 최소값
print(iris_normalized_df.max(), '\n')  # 정규화된 feature 들의 최대값
print(iris_normalized_df.mean(), '\n')  # 정규화된 feature 들의 평균값
print(iris_normalized_df.var(), '\n')  # 정규화된 feature 들의 분산값

"""
scaler 인스턴스를 통해 스케일링을 진행할 때 유의해야할 점

fit() 메소드는 데이터 변환을 위한 기준 정보 (데이터 세트의 최소, 최대값 등)을 설정하는 역할을 하고
transform() 메소드는 설정된 기준을 토대로 데이터를 변환한다.

즉, 테스트 데이터에 스케일링을 진행할 때 테스트 데이터 세트에 fit()을 수행한 결과로 transform()을 수행해야 한다.
그렇지 않으면 학습 데이터 세트와 테스트 데이터 세트의 기준 정보가 서로 달라진다.
"""
train_array = np.arange(0, 11).reshape(-1, 1)  # 2차원으로 reshape
test_array = np.arange(0, 6).reshape(-1, 1)

# fit : train, transform : train
scaler = MinMaxScaler()
scaler.fit(train_array)
train_scaled = scaler.transform(train_array)
print('원본 train set : ', train_array.reshape(-1))  # [ 0  1  2  3  4  5  6  7  8  9 10]
print('스케일 train set : ', train_scaled.reshape(-1))  # [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]

# fit : test, transform : test
scaler.fit(test_array)
test_scaled = scaler.transform(test_array)
print('원본 test set : ', test_array.reshape(-1))  # [0 1 2 3 4 5]
print('스케일 test set : ', test_scaled.reshape(-1))  # [0.  0.2 0.4 0.6 0.8 1. ]

# fit : train, transform : test
scaler.fit(train_array)
test_scaled = scaler.transform(test_array)
print('원본 test set : ', test_array.reshape(-1))  # [0 1 2 3 4 5]
print('스케일 test set : ', test_scaled.reshape(-1))  # [0.  0.1 0.2 0.3 0.4 0.5]
