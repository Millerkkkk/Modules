from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class GBClusterResult:
    gbs: List[List[int]]      # 局部索引 0..n-1
    centers: np.ndarray            # (k,2)
    radii: np.ndarray
