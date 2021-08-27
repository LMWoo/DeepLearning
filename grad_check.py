from __future__ import division
from __future__ import print_function
from os import tcgetpgrp

import torch

from torch.autograd import Variable, gradcheck

import lltm_cpp

lltm_cpp.forward()
# print(LLTMFunction.apply(*variables))

# if gradcheck(LLTMFunction.apply, variables):
#     print('OK')
import torch_test_cpp_extension.cpp as tc

x = torch.randn(4)
y = torch.randn(4)

print(tc.sigmoid_add(x, y))
