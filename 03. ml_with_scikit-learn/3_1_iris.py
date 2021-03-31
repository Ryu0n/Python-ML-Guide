import pandas as pd

from sklearn import datasets
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics
from sklearn.utils import Bunch

iris: Bunch = datasets.load_iris()
print(iris.keys())  # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

# 레이블, 레이블명
iris_target = iris.target
iris_target_names = iris.target_names
print('iris target', iris_target)
print('iris target name', iris_target_names)

# 데이터, 특징명
iris_data = iris.data
iris_feature_names = iris.feature_names
print('iris_data', iris_data)
print('iris_feature_names', iris_feature_names)

# 데이터와 특징명으로 데이터프레임 구성
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
print(iris_df.head(3))

# 데이터프레임에 레이블 컬럼 추가
iris_df['label'] = iris_target
print(iris_df.head(3))

# X : iris_data, y : iris_target
X_train, X_test, y_train, y_test = model_selection.train_test_split(iris_data, iris_target, test_size=0.2, random_state=11)

# Decision Tree 인스턴스 학습, 예측
dt_clf = tree.DecisionTreeClassifier(random_state=11)
dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, pred)  # 실제 정답, 예측 답 비교
print('정확도 : {0: 4f}'.format(accuracy))

