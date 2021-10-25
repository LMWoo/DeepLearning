# Project Purpose

 1. c++로 딥러닝 라이브러리를 구현하여 딥러닝관련 이해도와 c++기술을 향상시킨다.
 2. cuda로 딥러닝 라이브러리를 구현하여 병렬처리 프로그래밍과 gpu아키텍쳐를 이해한다.
 2. pytorch의 extension기능을 이용하여 python에서도 이용할 수 있게한다.
 3. 최근 많이 이용되는 PyTorch를 참고하여, PyTorch의 동작원리를 이해한다.

## Project Structure

* DeepLearning
  * cpp
    * cppRnn.hpp (cppTensor를 이용해 rnn구현)
    * cppTensor.hpp (Tensor기본 클래스)
    * cppTensor_Functions.hpp (cpu Tensor연산 구현, gpu는 cppTensor_gpu 함수들을 호출함)
    * cppTensor_gpu.cu (gpu Tensor연산 구현)
    * cppTensor_gpu.hpp (gpu 관련 함수 선언)
    * cppUtils.hpp (time check, exception check 등 구현)
    * main.cpp
  * python 
    * npRnn.py (rnn numpy로 구현)
    * setup.cpp (cppTensor, cppRnn extension c++ 소스)
    * setup.py (cppTensor, cppRnn extensions python 소스)
    * test.ipynb (rnn, lstm numpy로 구현)
    * test.py (npRnn.py, cppRnn.cpp test 소스, 현재 gpu 쪽에서 hidden_size가 32보다 클 경우 에러 발생)

# Installation

## Linux (20.04LTS)

### 소스 다운로드
```
git clone https://github.com/LMWoo/DeepLearning.git
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
894 elseif(USE_CUDA)
895  set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
896  if(CUDA_SEPARABLE_COMPILATION)
...
```

#### 수정 후
```
...
894 elseif(USE_CUDA)
895  list(APPEND Caffe2_GPU_SRCS {ProjectPath}/cpp/cppTensor_gpu.cu)
896  set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
897  if(CUDA_SEPARABLE_COMPILATION)
...
```

### PyTorch Build
```
git submodule sync
git submodule update --init --recursive --jobs 0
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```

# Running the test

### python extension 설치
```
cd python
python setup.py install
```

### test.py 실행
```
cd python
python test.py
```

# Requirements
 * python >= 3.6
 * cuda >= 11.2
 * cudnn >= 8.1.1
 * numpy >= 1.17.0
 * torchvision >= 0.2.1

# References
[pytorch](https://github.com/pytorch/pytorch) \
[numcpp](https://github.com/dpilger26/NumCpp) \
[cuda](http://www.kocw.or.kr/home/cview.do?cid=9495e57150084864)