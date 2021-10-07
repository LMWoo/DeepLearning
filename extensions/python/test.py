import torch
import numpy as np
import third_party_cpp as ncpp

x = torch.randn(4)
y = torch.randn(4)
print(ncpp.sigmoid_add(x, y))

shape = ncpp.Shape(5, 3)
cArray = ncpp.NdArray(shape)
ret = cArray.ones()
cArray = ncpp.zeros_like(cArray);
print(type(cArray));
print(type(cArray.getNumpyArray()));
print(cArray.getNumpyArray());
print(ncpp.toNumCpp(cArray.getNumpyArray()));

x = np.array([[1, 2]])
y = np.array([[2], [1]])
z = x @ y
print(z.shape)

x = ncpp.toNumCpp(x)
y = ncpp.toNumCpp(y)

print(ncpp.dot(x, y).getNumpyArray().shape)
