import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)

k_fold = KFold(n_splits=5)
cv_accuracy = []
print(features.shape)  # (150, 4)

n_iter = 0
for train_index, test_index in k_fold.split(features):
    # train : 학습, test : 검증
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]

    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    n_iter += 1

    accuracy = np.round(accuracy_score(y_test, pred), 4)
    cv_accuracy.append(accuracy)

print(cv_accuracy)
print(np.mean(cv_accuracy))