# Project Purpose

 1. c++로 딥러닝 라이브러리를 구현하여 딥러닝관련 이해도와 c++기술을 향상시킨다.
 2. cuda로 딥러닝 라이브러리를 구현하여 병렬처리 프로그래밍과 gpu아키텍쳐를 이해한다.
 2. pytorch의 extension기능을 이용하여 python에서도 이용할 수 있게한다.
 3. 최근 많이 이용되는 PyTorch를 참고하여, PyTorch의 동작원리를 이해한다.

# Project Progress

* DeepLearning
  * cpp
    * cppModules (Rnn, Linear 구현 완)
    * cppLoss (CrossEntropyLoss 구현 완)
    * cppOptimizer (Adagrad 구현 완)
    * cppTensor (tensor matrix multiply, add, transpose, activation functions..)
    * main.cpp
  * python (pybind11 사용)
    * test
      * rnn (numpy, cpu, gpu 구현 완)
      * lstm (numpy 구현 완)
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
|out = x @ y|out = cpp.matMul(x, y)|
|out = x + y|out = cpp.add(x, y)|
|out = np.tanh(x)|out = cpp.tanh(x)|
|out = np.exp(x)|out = cpp.exp(x)|

# cppModule Guide

## pytorch module version
```
class RNN(nn.Module):
  def __init__(...):
    self.rnn = nn.RNN(...)
    self.fc = nn.Linear(...)

  def forward(...):
    out = self.rnn(...)
    return self.fc(out)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

model = RNN(...)

outputs = model(...)
loss = criterion(outputs, labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## cppModule version
```

# module example
class RNN(nn.Module):
  def __init__(...):
    self.rnn = cpp.cppRnn(...)
    self.fc = cpp.cppLinear(...)

  def forward(...):
    out = self.rnn.forward(...)
    return self.fc.forward(out)

model = RNN(...)

criterion = cpp.CrossEntropyLoss()
optimizer = cpp.cppAdagrad(model.parameters(), learning_rate)

# train example
outputs = model.forward(...)
loss = criterion(outputs, labels)

optimizer.zero_grad()
criterion.backward(...)
optimizer.step()

```
 
# Experiments

## Environments
### Hardware information

* cpu - Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz
* gpu - GeForce GTX 1650 Ti 1.49 GHz

|name|size|
|----|----|
|VRAM|4GB|
|shared memory|48kb(per block)|
|warp size|32|
|Memory Bus Width|128bit|
|Multiprocessors|16|
|CUDA Cores|64(per MP)|

### Datasets

* MNIST

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
|gpu (not use sharedMemory) |95.08 %|10.89|
|gpu (use sharedMemory) |94.08 %|6.87|

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
