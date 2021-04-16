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

# print(iris_df, '\n')
# print(iris_df.mean(), '\n')  # feature 들의 평균값
# print(iris_df.var(), '\n')  # feature 들의 분산값

standard_scaler = StandardScaler()
standard_scaler.fit(iris_df)
iris_scaled = standard_scaler.transform(iris_df)
iris_scaled_df = pd.DataFrame(data=iris_scaled, columns=iris_feature_names)

# print(iris_scaled_df, '\n')
# print(iris_scaled_df.mean(), '\n')  # 스케일링된 feature 들의 평균값
# print(iris_scaled_df.var(), '\n')  # 스케일링된 feature 들의 분산값


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

# 학습 데이터 세트 스케일링
train_array = np.arange(0, 11).reshape(-1, 1)  # 2차원으로 reshape
test_array = np.arange(0, 6).reshape(-1, 1)

scaler = MinMaxScaler()
scaler.fit(train_array)
train_scaled = scaler.transform(train_array)
"""
ValueError: Expected 2D array, got 1D array instead:
array=[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10.].
"""

print('원본 train set : ', train_array.reshape(-1))
print('스케일 train set : ', train_scaled.reshape(-1))

# 잘못된 테스트 데이터 세트 스케일링
scaler.fit(test_array)
test_scaled = scaler.transform(test_array)

print('원본 test set : ', test_array.reshape(-1))
print('스케일 test set : ', test_scaled.reshape(-1))

# 올바른 테스트 데이터 세트 스케일링 (학습 데이터 세트의 fit()으로 테스트 데이터 세트 transform())
scaler.fit(train_array)
test_scaled = scaler.transform(test_array)

print('원본 test set : ', test_array.reshape(-1))
print('스케일 test set : ', test_scaled.reshape(-1))
