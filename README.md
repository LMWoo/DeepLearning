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

#### 수정 전
```
...
      if(NOT NO_API)
        add_subdirectory(${TORCH_ROOT}/test/cpp/api ${CMAKE_BINARY_DIR}/test_api)
      endif()
...
```

### 수정 후
```
...
      if(NOT NO_API)
        add_subdirectory(${TORCH_ROOT}/test/cpp/api ${CMAKE_BINARY_DIR}/test_api)
        add_subdirectory(${TORCH_ROOT}/../MyTorch/cpp ${CMAKE_BINARY_DIR}/MyTorch_cpp)
      endif()
...
```

### PyTorch 빌드
```
git submodule sync
git submodule update --init --recursive --jobs 0
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```

# References
https://github.com/pytorch/pytorch
