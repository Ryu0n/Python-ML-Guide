import pandas as pd
import numpy as np

"""
index : RDBMS의 PK처럼 개별 데이터를 고유하게 식별하는 key값
Series와 DataFrame 모두 index를 key값으로 가지고 있다.
Series와 DataFrame의 차이점 : Series는 컬럼이 하나, DataFrame은 컬럼이 여러개 
즉, DataFrame은 여러개의 Series로 구성되어 있다고 봐도 무방하다.
"""

# read_csv() 메소드의 sep인자에 구분자를 넣을 수 있다.
# 예를 들어, csv 파일같은 경우는 콤마(,)로 구분되어 있지만
# tsv 파일은 탭('\t')으로 구분되어 있기 때문에 read_csv(sep='\t')를 호출하면 된다.

titanic_df = pd.read_csv('../datas/titanic_train.csv')
print(type(titanic_df))
print(titanic_df.head(3))
print(titanic_df.head(3).shape)
print(titanic_df.info())
print(titanic_df.describe())  # 25% (0~25% 백분위수), 50(25~50%), .. / 50백분위수는 중앙값과 같다

pclass_series = titanic_df['Pclass']  # Serires 인스턴스
print(pclass_series.head())  # seires의 인덱스, series의 데이터값
print(pclass_series.value_counts())  # 각 도메인의 갯수 (분포도 파악에 유리하다.)
print(pclass_series.value_counts().head(1))

# 컬렉션 -> DataFrame (1)
col_name1 = ['col1']
list1 = [1, 2, 3]
array1 = np.array(list1)
df_list1 = pd.DataFrame(data=list1, columns=col_name1)
df_array1 = pd.DataFrame(data=array1, columns=col_name1)
print('df_list : \n', df_list1)
print('df_array : \n', df_array1)

# 컬렉션 -> DataFrame (2)
col_name2 = [
    'col1',
    'col2',
    'col3'
]
list2 = [[1, 2, 3],
         [11, 12, 13]]
df_array2 = pd.DataFrame(data=list2, columns=col_name2)
print('df_array2 : \n', df_array2)

# 컬렉션 -> DataFrame (3)
dictionary = {'col1': [1, 11], 'col2': [2, 22], 'col3': [3, 33]}
df_dict = pd.DataFrame(dictionary)
print('df_dict : \n', df_dict)

# DataFrame -> 컬렉션 (1)
array3 = df_dict.values
list3 = array3.tolist()
dict3 = df_dict.to_dict()
print(array3, list3)
print(dict3)
print(df_dict.axes)

# 새로운 칼럼 추가
titanic_df['Age_0'] = 0
print(titanic_df.head(3))
# 기존 DataFrame의 Series를 통해 새로운 칼럼 추가
titanic_df['Age_by_10'] = titanic_df['Age'] * 10
titanic_df['Family_No'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1
print(titanic_df.head(3))
# 브로드캐스팅 연산
titanic_df['Age_by_10'] += 100
print(titanic_df.head(3))

# DataFrame 데이터삭제
# DataFrame.drop()
# - axis=0 : (열방향)행 삭제 / axis=1 : (행방향)컬럼 삭제
# - inplace=True : immutable (자기자신 수정) / inplace=False : mutable (자기자신 수정 x)
titanic_drop_df = titanic_df.drop('Age_0', axis=1)
print(titanic_drop_df.head(3))
# inplace=True 일 경우 mutable 이므로 None 반환, 원본 인스턴스에 변화
drop_result = titanic_df.drop(['Age_0', 'Age_by_10', 'Family_No'], axis=1, inplace=True)
print(drop_result, '\n', titanic_df)  # None, DataFrame


