import os
import sys
import torch.cuda
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

setup(
    name='third_party_cpp',
    include_dirs='../../cpp/third_party',
    ext_modules=[
    CppExtension(
        'third_party_cpp', ['setup.cpp']),
    ],
    cmdclass={'build_ext': BuildExtension})