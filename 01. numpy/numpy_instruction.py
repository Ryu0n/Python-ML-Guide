import numpy as np

array1 = np.array([1, 2, 3])
array2 = np.array([[1, 2, 3],
                   [2, 3, 4]])

print(array1, type(array1), array1.shape)
print(array2, type(array2), array2.shape)

# 리스트를 ndarray로 캐스팅 가능
li = [1, 2, 3]
array3 = np.array(li)
print(array3, type(array3), array3.shape)

# 데이터 타입
print(array3.dtype)

# 데이터 타입 캐스팅 가능
array_float = array3.astype('float64')
print(array_float, type(array_float), array_float.shape, array_float.dtype)

sequence_array = np.arange(10)
print(sequence_array)

zero_array = np.zeros((3, 2), dtype='int32')
one_array = np.ones((4, 3), dtype='int32')
print(zero_array, '\n', one_array)

reshaped_array1 = sequence_array.reshape(2, 5)
reshaped_array2 = sequence_array.reshape(5, 2)
print(reshaped_array1, '\n', reshaped_array2)
# -1을 reshape인자로 넣을 경우 가능한 최대의 행 혹은 열로 확장된다.
reshaped_array3 = sequence_array.reshape(-1, 5)
reshaped_array4 = sequence_array.reshape(5, -1)

# numpy array를 list로 캐스팅
list3 = reshaped_array3.tolist()
print(list3, type(list3))

# 3차원 array
array = np.arange(8)
array3d = array.reshape((2, 2, 2))
print(array3d, array3d.shape)

# 3차원 행렬을 2차원으로 변환
array2d = array3d.reshape((-1, 1))
print(array2d, array2d.shape)

# 인덱싱
array1 = np.arange(start=1, stop=10)
print(array1, array1[0], array1[-1])
array1[0] = 9
print(array1)

array2d = array1.reshape((3, 3))
print(array2d, array2d[0, 2], array2d[1, 1], array2d[2, 0])

# 슬라이싱
print(array1[0:3])
print(array1[:3], array1[3:])
print(array2d[1])
print(array2d[0:2, 1:2])

# 팬시 인덱싱
print(array2d[0:2])  # 인덱싱
print(array2d[[0, 1]])  # 팬시 인덱싱
print(array2d[[0, 1], 1:2])  # 0, 1행의 1번째 열

# 불린 인덱싱
print(array1 > 5, type(array1 > 5))
print(array1[array1 > 5])

# True에 해당하는 요소만 출력
print(array1)  # [9 2 3 4 5 6 7 8 9]
bool_indexes = array1 > 5
print(array1[bool_indexes])  # [9 6 7 8 9]
# 0번째 요소만 출력, 나머지 요소는 False
indexes = np.array([0])
print(array1[indexes])  # [9]
# 0, 2, 4번째 요소만 출력
indexes = np.array([0, 2, 4])
print(array1[indexes])  # [9 3 5]

# 행렬 정렬 - np.sort() : immutable
org_array = np.array([3, 1, 9, 5])
sort_array = np.sort(org_array)
print(org_array, sort_array)
# 행렬 정렬 - ndarray.sort() : mutable
org_array.sort()
print(org_array)
# 역정렬
reversed_array = np.sort(org_array)[::-1]
print(reversed_array)
# 2차원 행렬 정렬
array2d = np.array([[8, 12],
                    [7, 1]])
sorted_array_axis0 = np.sort(array2d, axis=0)  # 열 기준 정렬
sorted_array_axis1 = np.sort(array2d, axis=1)  # 행 기준 정렬
print(sorted_array_axis0)
print(sorted_array_axis1)

# argsort - 원본 행렬이 정렬되었을 때, 원본행렬의 원소에 대한 인덱스가 필요할 경우
org_array = np.array([3, 1, 9, 5])
# 작은 값부터 순서대로 데이터의 index를 반환해줌
sort_indices = np.argsort(org_array)  # 오름차순으로 정렬된 인덱스 [1 0 3 2]
"""
과정
1. [3, 1, 9, 5] -> [0, 1, 2, 3] 인덱스를 가지고 있음
2. [3, 1, 9, 5] -> [1, 3, 5, 9] 오름차순으로 정렬
3. [1, 3, 5, 9] -> [1, 0, 3, 2] 정렬된 각 원소를 원본배열로부터의 인덱스로 치환
"""

# 큰 값부터 순서대로 데이터의 index를 반환해줌
sort_indices_desc = np.argsort(org_array)[::-1]  # 내림차순으로 정렬된 인덱스 [2 3 0 1]

print(sort_indices)
print(sort_indices_desc)

name_array = np.array(['John', 'Mike', 'Sarah', 'Kate', 'Samuel'])
score_array = np.array([78, 95, 84, 98, 88])
sort_indices_asc = np.argsort(score_array)

print(sort_indices_asc)  # [0 2 4 1 3]
print(name_array[sort_indices_asc])  # fancy indexing : ['John' 'Sarah' 'Samuel' 'Mike' 'Kate']
