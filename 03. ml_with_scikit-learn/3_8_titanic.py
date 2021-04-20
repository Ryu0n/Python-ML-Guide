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
print(titanic_df['Cabin'].head(3))
print(titanic_df.groupby(['Sex', 'Survived'])['Survived'].count())

# 성별에 따른 생존률
sns.barplot(x='Sex', y='Survived', data=titanic_df)
plt.show()

# Pclass : 객실
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df)
plt.show()
