# About MyTorch

cpp, python(extension)으로 딥러닝 라이브러리를 직접 구현
실제 라이브러리 pytorch와 비교 

# Installation

## Linux (20.04LTS)

### MyTorch Source Download
```
git clone MyTorch
cd MyTorch
```

### PyTorch Source Download
```
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
```

### PyTorch 빌드하기 전 Caffe2 CMakeLists.txt수정

### Build PyTorch From Source
git submodule sync
git submodule update --init --recursive --jobs 0
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```

# References
https://github.com/pytorch/pytorch
