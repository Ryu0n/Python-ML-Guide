import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', None)

iris = load_iris()
data = iris.data
label = iris.target

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=121)

dt_clf = DecisionTreeClassifier()

hyper_params = {'max_depth': [1, 2, 3],
                'min_samples_split': [2, 3]}

grid_dt_clf = GridSearchCV(estimator=dt_clf, param_grid=hyper_params, scoring='accuracy', cv=3)
grid_dt_clf.fit(X_train, y_train)

scores_df = pd.DataFrame(grid_dt_clf.cv_results_)
# print(scores_df.columns)
# print(grid_dt_clf.cv_results_)
# print(scores_df)
print(scores_df[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score']])
