import sys
import torch.cuda
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

setup(
    name='torch_test_cpp_extension',
    ext_modules=[
    CppExtension(
        'torch_test_cpp_extension.cpp', ['extension.cpp']),
    ],
    cmdclass={'build_ext': BuildExtension})
