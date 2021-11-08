import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import cpp as cpp

print("numpy transpose")
x = np.random.randn(2, 3, 4)
x_t = x.transpose((2, 1, 0))

print(x_t)

print("vec3 transpose")
x = cpp.cppTensorVec3(x)
x_t = cpp.transpose(x, (2, 1, 0))

print(x_t.numpy())