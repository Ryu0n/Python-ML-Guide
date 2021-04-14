"""
ML 알고리즘의 데이터는 문자열 값을 허용하지 않는다. 즉, 숫자형 데이터로 인코딩해야한다.
데이터 인코딩에는 크게 두 가지 방법이 있다.

1. 레이블 인코딩 (Label Encoding)
2. 원-핫 인코딩 (One-hot Encoding)

레이블 인코딩은 문자열 iterator를 순서대로 숫자로 매핑한다.
하지만 이로인해 특정 알고리즘에서는 성능이 떨어진다. 그 이유는 숫자가 오름차순으로 매핑됨에 의해
가중치가 부여되는 현상이 발생하기 때문이다. 이는 단순 코드이지, 중요도로 인식되서는 안된다. (트리계열 알고리즘은 OK)
고로 이러한 문제를 해결하기 위한 방식은 원-핫 인코딩이다.

원-핫 인코딩은 새로운 피처를 추가하여 해당하는 컬럼에만 1, 나머지는 0으로 표시하는 방식이다.
ex)
        상품분류_TV    상품분류_냉장고    상품분류_전자렌지
TV          1             0               0
냉장고       0             1               0
전자렌지     0             0               1

원-핫 인코더는 변환하기 전 모든 문자열 값이 숫자형으로 변환되어야 한다.
원-핫 인코더는 입력값이 2차원 데이터가 필요하다.
"""

# Label Encoding
from sklearn.preprocessing import LabelEncoder
# One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder
import numpy as np

items = ['TV', '냉장고', '전자레인지', '컴퓨터']

encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print(labels)  # [0 1 2 3]
print(encoder.classes_)  # ['TV' '냉장고' '전자레인지' '컴퓨터']
print(encoder.inverse_transform(labels))  # ['TV' '냉장고' '전자레인지' '컴퓨터']

# 먼저 숫자 값으로 변환을 위해 LabelEncoder 사용
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
labels = labels.reshape(-1, 1)  # 2차원 데이터로 변환
# One-Hot Encoder 사용
oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)
print(oh_labels)
print(type(oh_labels))
print(oh_labels.shape)
print(oh_labels.toarray())

# 판다스에는 원-핫 인코딩을 더 쉽게 지원하는 API가 있다.
import pandas as pd
df = pd.DataFrame({'item': ['TV', '냉장고', '전자레인지', '컴퓨터']})
print(df)
oh_df = pd.get_dummies(df)
print(oh_df)
