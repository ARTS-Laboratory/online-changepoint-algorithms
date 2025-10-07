from typing import Optional

import numpy as np

def generate_normal(mean: float, std_dev: float, num: int, seed: Optional[int] = None):
    """ """
    rng = np.random.default_rng(seed=seed)
    return rng.normal(mean, std_dev, num)