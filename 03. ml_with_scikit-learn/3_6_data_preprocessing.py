"""
ML 알고리즘의 데이터는 문자열 값을 허용하지 않는다. 즉, 숫자형 데이터로 인코딩해야한다.
데이터 인코딩에는 크게 두 가지 방법이 있다.

1. 레이블 인코딩 (Label Encoding)
2. 원-핫 인코딩 (One-hot Encoding)

레이블 인코딩은 문자열 iterator를 순서대로 숫자로 매핑한다.
하지만 이로인해 특정 알고리즘에서는 성능이 떨어진다. 그 이유는 숫자가 오름차순으로 매핑됨에 의해
가중치가 부여되는 현상이 발생하기 때문이다. 이는 단순 코드이지, 중요도로 인식되서는 안된다. (트리계열 알고리즘은 OK)
고로 이러한 문제를 해결하기 위한 방식은 원-핫 인코딩이다.
"""

# Label Encoding
from sklearn.preprocessing import LabelEncoder

items = ['TV', '냉장고', '전자레인지', '컴퓨터']

encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print(labels)  # [0 1 2 3]
print(encoder.classes_)  # ['TV' '냉장고' '전자레인지' '컴퓨터']
print(encoder.inverse_transform(labels))  # ['TV' '냉장고' '전자레인지' '컴퓨터']
