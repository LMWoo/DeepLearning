# 프로젝트 소개

 1. c++로 딥러닝 라이브러리를 직접 구현한다.
 2. pytorch의 extension기능을 이용하여 python에서도 이용할 수 있게한다.
 3. 최근 많이 이용되는 PyTorch를 참고하여, PyTorch의 동작원리를 이해한다.
 4. c++로 딥러닝 라이브러리를 구현하여 딥러닝관련 이해도와 c++기술을 향상시킨다.

## 프로젝트 구조

* DeepLearning
  * pytorch
  * cpp
    * main.cpp
  * python 
    * extensions
    * example.py

# Installation

## Linux (20.04LTS)

### 소스 다운로드
```
git clone DeepLearning
cd DeepLearning
```

### PyTorch 소스 다운로드
```
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
```

### PyTorch 빌드하기 전 Caffe2의 CMakeLists.txt파일 수정

#### 수정 전
```
...
1107      if(NOT NO_API)
1108        add_subdirectory(${TORCH_ROOT}/test/cpp/api ${CMAKE_BINARY_DIR}/test_api)
1109      endif()
...
```

#### 수정 후
```
...
1107      if(NOT NO_API)
1108        add_subdirectory(${TORCH_ROOT}/test/cpp/api ${CMAKE_BINARY_DIR}/test_api)
1109        add_subdirectory(${TORCH_ROOT}/../DeepLearning/cpp ${CMAKE_BINARY_DIR}/DeepLearning_cpp)
1110      endif()
...
```

### PyTorch 빌드
```
git submodule sync
git submodule update --init --recursive --jobs 0
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```

# Running the example (starting from DeepLearning folder)
## c++
```
cd pytorch
cd build
cd bin
./DeepLearning_cpp
```

## python

### python 확장 설치
```
cd python
cd extensions
python setup.py install
```

### example.py 실행
```
cd python
python example.py
```

# Requirements
 * python >= 3.6
 * cuda >= 11.2
 * cudnn >= 8.1.1

# References
[pytorch](https://github.com/pytorch/pytorch) \
[numcpp](https://github.com/dpilger26/NumCpp)
