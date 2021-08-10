import ctypes
import builtins
from torch import *

clibs = ctypes.CDLL("/media/lee/ESD-ISO/study/MyTorch/build/libMyTorch.so", mode=ctypes.RTLD_GLOBAL)

path = "/media/lee/ESD-ISO/study/no_pytorch_study/torch/bin/torch_shm_manager"
# _initExtension(path.encode('utf-8'))
# print(is_grad_enabled())
# clibs.initModule()
print(clibs.is_prime(3))

print(path.encode('utf-8'))