import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


titanic_df = pd.read_csv('../datas/titanic_train.csv')
print(titanic_df.head(3))
print(titanic_df.info())

print(titanic_df['Age'])
print('here \n', titanic_df['Cabin'])
print(titanic_df['Embarked'])

titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)
titanic_df['Cabin'].fillna('N', inplace=True)
titanic_df['Embarked'].fillna('N', inplace=True)

print(titanic_df.isnull())
print(titanic_df.isnull().sum())
print(titanic_df.isnull().sum().sum())

print(titanic_df['Sex'].value_counts())
print(titanic_df['Cabin'].value_counts())
print(titanic_df['Embarked'].value_counts())

titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]  # 문자열의 앞에 한글자만 추출
