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
x.cuda()
y = cpp.cppTensorVec3(y)
y.cuda()
result = cpp.matMul_vec3(x, y)
result.cpu()
print(result.numpy())

print("conv")

output_z = 6
input_z = 3
input_y = 7
input_x = 8
kernel_size = 5

x = np.random.randn(input_z, input_y, input_x)
i = torch.Tensor(x).unsqueeze(0)
c = nn.Conv2d(input_z, output_z, kernel_size=5, stride=1, padding=0)
o = c(i)
print(o[0][0])

class cppCNN(object):
    def __init__(self, in_num, out_num):
        w = c.weight.detach().numpy()
        self.weight = [cpp.cppTensorVec3(w[i]) for i in range(out_num)]
        self.out_num = out_num
        self.in_num = in_num
        
    def __call__(self, x):
        out = []
        for i in range(self.out_num):
            out.append(cpp.conv_vec3(x, self.weight[i]))
        return np.array(out)

x = cpp.cppTensorVec3(x)
c = cppCNN(input_z, output_z)
result = c(x)
print(result[0].numpy())

print('end')