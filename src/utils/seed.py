import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, PyTorch (CPU and CUDA) for full reproducibility.

    Args:
        seed: The integer seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic CUDA algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
