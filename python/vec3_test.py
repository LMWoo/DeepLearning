import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import cpp as cpp

print("numpy permute")
x = np.random.randn(2, 6, 2)
x_t = x.transpose((1, 2, 0))

print(x_t)

print("vec3 permute")
x = cpp.cppTensorVec3(x)
x.cuda()
x_t = cpp.permute_vec3(x, (1, 2, 0))
x_t.cpu()
print(x_t.numpy())

print("numpy matmul")
x = np.random.randn(2, 4, 2)
y = np.random.randn(2, 2, 4)
print(x @ y)

print("vec3 matmul")
x = cpp.cppTensorVec3(x)
y = cpp.cppTensorVec3(y)
result = cpp.matMul_vec3(x, y)
print(result.numpy())