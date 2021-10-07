import torch
import third_party_cpp as test_cpp

x = torch.randn(4)
y = torch.randn(4)
print(test_cpp.sigmoid_add(x, y))

shape = test_cpp.Shape(5, 3)
cArray = test_cpp.NdArray(shape)
ret = cArray.ones()
cArray = test_cpp.zeros_like(cArray);
print(type(cArray));
print(type(cArray.getNumpyArray()));
print(cArray.getNumpyArray());
print(test_cpp.toNumCpp(cArray.getNumpyArray()));