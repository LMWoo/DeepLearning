import builtins
_bool = builtins.bool

def _initExtension(shm_manager_path: str) -> None: ...  # THPModule_initExtension
def is_grad_enabled() -> bool: ...