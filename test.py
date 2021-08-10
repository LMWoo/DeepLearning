import ctypes

def _initExtension(shm_manager_path: str) -> None: ...  # THPModule_initExtension

ctypes.CDLL("/media/lee/ESD-ISO/study/MyTorch/build/libMyTorch.so", mode=ctypes.RTLD_GLOBAL)

path = "/media/lee/ESD-ISO/study/no_pytorch_study/torch/bin/torch_shm_manager"
_initExtension(path.encode('utf-8'))
print(path.encode('utf-8'))