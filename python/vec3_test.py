import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import cpp as cpp

print("numpy permute")
x = np.random.randn(2, 3, 4)
x_t = x.transpose((1, 2, 0))

print(x_t)

print("vec3 permute")
x = cpp.cppTensorVec3(x)
x_t = cpp.permute(x, (1, 2, 0))

print(x_t.numpy())