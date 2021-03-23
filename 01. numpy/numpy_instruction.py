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
