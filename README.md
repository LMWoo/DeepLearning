# Project Purpose

 1. c++로 딥러닝 라이브러리를 구현하여 딥러닝관련 이해도와 c++기술을 향상시킨다.
 2. cuda로 딥러닝 라이브러리를 구현하여 병렬처리 프로그래밍과 gpu아키텍쳐를 이해한다.
 2. pytorch의 extension기능을 이용하여 python에서도 이용할 수 있게한다.
 3. 최근 많이 이용되는 PyTorch를 참고하여, PyTorch의 동작원리를 이해한다.

# Project Progress

* DeepLearning
  * cpp
    * cppNN (RNN 구현 완)
    * cppTensor (tensor matrix multiply, add, transpose, activation functions..)
    * main.cpp
  * python (pybind11 사용)
    * test
      * rnn (numpy, cpu, gpu 구현 완)
      * lstm (numpy 구현 )
      * gru
    * setup.cpp
    * setup.py

# cppTensor Guide

## Basic
|numpy|cppTensor|
|----|----|
|x = np.array((2, 3))|x = cpp.cppTensor(2, 3)|
||y = cpp.cppTensor(numpy array)|
|x.zeros()|x.zeros()|
|x.ones()|x.ones()|
|y = x.T|y = cpp.transpose(x)|

## Calculation
|numpy|cppTensor|
|----|----|
|out = x @ y|out = cpp.matMul<double>(x, y)|
|out = x + y|out = cpp.add<double>(x, y)|
|out = np.tanh(x)|out = cpp.tanh<double>(x)|
|out = np.exp(x)|out = cpp.exp<double>(x)|
 
 # cppRNN Guide
|pytorch|cppRNN|
|----|----|
|model = nn.RNN(input_size, hidden_size)|model = cpp.cppRnn(..., input_size, hidden_size, num_classes)|
|criterion = nn.CrossEntropyLoss()||
|optimizer = optim.Adam(model.parameters(), lr=0.01)||
|outputs = model.forward(x, hprev)|outputs = model.forward(x, hprev)|
|loss = criterion(outputs, targets)|dy, loss = model.cross_entropy_loss(outputs)|
|gradients = model.backward(dy)|gradients = model.backward(dy)|
|optimizer.step()|model.optimizer(gradients)|
 
# Experiments

## Hardware information
* cpu - Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz
* gpu - GeForce GTX 1650 Ti 1.49 GHz, VRAM 4GB, shared memory 48kb(per block), warp size 32, Memory Bus Width 128bit

## RNN Results
|Hyper Parameters|value|
|----|----|
|seq_length|28|
|input_size|28|
|num_classes|10|
|hidden_size|256|
|learning_rate|0.01|

|condition|Accuracy|Speed (s / 1000 images)|
|----|----|----|
|cpu |95.00 %|20.47|
|gpu (TILED_WIDTH = 32, not use sharedMemory) |94.95 %|10.89|
|gpu (TILED_WIDTH = 8, not use sharedMemory) |94.08 %|4.47|
|gpu (TILED_WIDTH = 8, use sharedMemory) |94.08 %|4.87|

# Note

* [CUDA](note/CUDA.pdf)
* [DeepLearning](note/DeepLearning.pdf)

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
895  list(APPEND Caffe2_GPU_SRCS {ProjectPath}/cpp/cppTensor/cppTensor_gpu.cu)
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
```
cd python
python setup.py install
```

* rnn
```
python test/rnn/np_test.py (numpy version)
python test/rnn/cpu_test.py (cpu version)
python test/rnn/gpu_test.py (gpu version)
```


* lstm
```
python test/lstm/np_test.py (numpy version)
python test/lstm/cpu_test.py (cpu version)
python test/lstm/gpu_test.py (gpu version)
```

* gru
```
python test/gru/cpu_test.py (cpu version)
python test/gru/gpu_test.py (gpu version)
```

# Requirements
 * python >= 3.6
 * cuda >= 11.2
 * cudnn >= 8.1.1
 * numpy >= 1.17.0
 * torchvision >= 0.2.1

# References
 * [pytorch](https://github.com/pytorch/pytorch)
 * [pytorch-cpp](https://github.com/prabhuomkar/pytorch-cpp)
 * [numcpp](https://github.com/dpilger26/NumCpp)
 * [cuda](http://www.kocw.or.kr/home/cview.do?cid=9495e57150084864)
