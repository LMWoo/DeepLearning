# About MyTorch

cpp, python(extension)으로 딥러닝 라이브러리를 직접 구현
실제 라이브러리 pytorch와 비교 

# Installation

## Linux (20.04LTS)

### 소스 다운로드
```
git clone MyTorch
cd MyTorch
```

### PyTorch 소스 다운로드
```
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
```

### PyTorch 빌드하기 전 Caffe2 CMakeLists.txt수정
```
...

      if(USE_DISTRIBUTED)
        add_subdirectory(${TORCH_ROOT}/test/cpp/c10d ${CMAKE_BINARY_DIR}/test_cpp_c10d)
        if(NOT WIN32)
          add_subdirectory(${TORCH_ROOT}/test/cpp/dist_autograd ${CMAKE_BINARY_DIR}/dist_autograd)
          add_subdirectory(${TORCH_ROOT}/test/cpp/rpc ${CMAKE_BINARY_DIR}/test_cpp_rpc)
        endif()
      endif()
      if(NOT NO_API)
        add_subdirectory(${TORCH_ROOT}/test/cpp/api ${CMAKE_BINARY_DIR}/test_api)
      endif()
...
```

```
...

      if(USE_DISTRIBUTED)
        add_subdirectory(${TORCH_ROOT}/test/cpp/c10d ${CMAKE_BINARY_DIR}/test_cpp_c10d)
        if(NOT WIN32)
          add_subdirectory(${TORCH_ROOT}/test/cpp/dist_autograd ${CMAKE_BINARY_DIR}/dist_autograd)
          add_subdirectory(${TORCH_ROOT}/test/cpp/rpc ${CMAKE_BINARY_DIR}/test_cpp_rpc)
        endif()
      endif()
      if(NOT NO_API)
        add_subdirectory(${TORCH_ROOT}/test/cpp/api ${CMAKE_BINARY_DIR}/test_api)
        add_subdirectory(${TORCH_ROOT}/../MyTorch/cpp ${CMAKE_BINARY_DIR}/MyTorch_cpp)
      endif()
...
```

### Build PyTorch From Source
git submodule sync
git submodule update --init --recursive --jobs 0
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```

# References
https://github.com/pytorch/pytorch
