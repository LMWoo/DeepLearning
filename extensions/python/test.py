import torch
import numpy as np
import third_party_cpp as ncpp

# x = torch.randn(4)
# y = torch.randn(4)
# print(ncpp.sigmoid_add(x, y))

# shape = ncpp.Shape(5, 3)
# cArray = ncpp.NdArray(shape)
# ret = cArray.ones()
# cArray = ncpp.zeros_like(cArray);
# print(type(cArray));
# print(type(cArray.getNumpyArray()));
# print(cArray.getNumpyArray());
# print(ncpp.toNumCpp(cArray.getNumpyArray()));

def printResult(x, y):
    print('numpy')
    print(x @ y)

    x = ncpp.NdArray(x)
    y = ncpp.NdArray(y)
    print('numcpp')
    print(ncpp.dot(x, y).getNumpyArray())

x = np.random.randn(5, 3)
xx = ncpp.NdArray(x)
xx.print()
ncpp.memoryFree(xx)

ncpp.test_gpu()
ncpp.test_gpu_matrix_add()

# x = np.random.randn(5, 3)
# y = np.random.randn(3, 5)
# xx = ncpp.NdArray(x)
# yy = ncpp.NdArray(y)
# result = ncpp.dot(xx, yy)
# result.print()

# x = np.random.randn(1, 1)
# y = np.random.randn(1, 1)

# printResult(x, y)

# x = np.random.randn(1, 5)
# y = np.random.randn(5, 3)

# printResult(x, y)