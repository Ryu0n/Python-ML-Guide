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
