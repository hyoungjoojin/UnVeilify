from typing import Union

import numpy as np
import torch

__all__ = ["set_seed"]


def set_seed(seed: Union[int, None]) -> None:
    if seed is None:
        return

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
