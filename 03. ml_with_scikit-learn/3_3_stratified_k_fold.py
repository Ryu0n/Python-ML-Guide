import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
features = iris.data
label = iris.target
print(features, label)

stk_fold = StratifiedKFold(n_splits=3)
dt_clf = DecisionTreeClassifier(random_state=156)
cv_accuracy = list()

for train_index, test_index in stk_fold.split(features, label):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]

    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    cv_accuracy.append(accuracy)

print(np.mean(cv_accuracy))

# iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# iris_df['label'] = iris.target
# print(iris_df['label'].value_counts())
# print(iris_df['label'])

# k_fold = KFold(n_splits=3)
# for train_index, test_index in k_fold.split(iris_df):
#     print(train_index, test_index)
#     print(iris_df['label'].iloc[train_index])
#     print(iris_df['label'].iloc[test_index])

# k_fold = StratifiedKFold(n_splits=3)
# for train_index, test_index in k_fold.split(iris_df, iris_df['label']):
#     print(train_index, test_index)
#     print(iris_df['label'].iloc[train_index])
#     print(iris_df['label'].iloc[test_index])
