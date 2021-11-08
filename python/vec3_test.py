import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import cpp as cpp

print("numpy permute")
x = np.random.randn(6, 960, 1024)
x_t = x.transpose((1, 2, 0))

print(x_t)

print("vec3 permute")
x = cpp.cppTensorVec3(x)
x.cuda()
x_t = cpp.permute(x, (1, 2, 0))
x_t.cpu()
print(x_t.numpy())