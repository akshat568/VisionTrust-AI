import os
import random
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, PyTorch (CPU and CUDA) for full reproducibility.

    Args:
        seed: The integer seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
