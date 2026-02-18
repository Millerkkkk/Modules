from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

from .node import Node


def compute_distance_matrix(nodes) -> np.ndarray:
    n = len(nodes)
    dist = np.zeros((n, n), dtype=float)

    for i in range(n):
        xi, yi = nodes[i].x, nodes[i].y
        for j in range(n):
            dist[i, j] = math.hypot(xi - nodes[j].x,
                                    yi - nodes[j].y)

    return dist



def submatrix(dist_all: Sequence[Sequence[float]], indices: Sequence[int]) -> List[List[float]]:
    """从全距离矩阵抽取子矩阵"""
    return [[dist_all[i][j] for j in indices] for i in indices]

