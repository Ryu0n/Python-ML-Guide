"""
cross_val_score(estimator, X, y=None, scoring=None, cv=None, n_jobs=1, verbose=0, fit_params=none, pre_dispatch='2*n_jobs')

1. 폴드 세트를 설정
2. for 루프에서 반복으로 학습 및 테스트 데이터의 인덱스 추출
3. 반복적으로 학습, 예측을 수행하고 예측 성능 반환
위의 과정을 한번에 수행

classifier 가 입력되면 Stratified K 폴드 방식으로 레이블값의 분포에 따라 학습/테스트 세트를 분할

estimator : classifier, regressor
X : feature data set
y : label data set
scoring : 예측 성능 평가 지표
cv : 교차 검증 폴드 수

:return: scoring 파라미터로 지정된 성능 지표 측정값 (배열)
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.datasets import load_iris

iris_data = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)

data = iris_data.data
label = iris_data.target

scores = cross_val_score(dt_clf, data, label, scoring='accuracy', cv=3)
print(scores)  # 폴드별 정확도
print(np.round(scores, 4))  # 교차 검증별 정확도
print(np.round(np.mean(scores), 4))  # 평균 검증 정확도
