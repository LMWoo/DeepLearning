import torch
import torch_test_cpp_extension.cpp as test_cpp

x = torch.randn(4)
y = torch.randn(4)
print(test_cpp.sigmoid_add(x, y))