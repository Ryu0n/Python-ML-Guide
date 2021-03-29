import pandas as pd

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


