import os
import sys
import torch.cuda
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

setup(
    name='cpp',
    include_dirs=['/usr/local/cuda/include', '../cpp'],
    ext_modules=[
    CUDAExtension(
        name='cpp',
            sources=[
                'setup.cpp',
                '../cpp/cppTensor/cppTensor_gpu.cu'])
            ],
    cmdclass={'build_ext': BuildExtension})