import os
import sys
import torch.cuda
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

setup(
    name='torch_test_cpp_extension',
    include_dirs='/media/lee/ESD-ISO/study/DeepLearning/cpp/third_party',
    ext_modules=[
    CppExtension(
        'torch_test_cpp_extension.cpp', ['setupCpp.cpp']),
    ],
    cmdclass={'build_ext': BuildExtension})