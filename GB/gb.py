from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class GranularBall:
    """
    通用 GranularBall：给定点索引集合，计算中心与半径
    - points_idx: 点在原 points 数组中的索引（局部索引 0..n-1）
    - data: points ndarray, shape (n,2)
    """
    points_idx: List[int]
    data: np.ndarray

    def __post_init__(self) -> None:
        self.points_idx = list(map(int, self.points_idx))
        self.data = np.asarray(self.data)
        self._center_radius()

    def _center_radius(self) -> None:
        """根据 points_idx 重新计算 center / radius"""
        if self.points_idx:
            pts = self.data[self.points_idx]
            self.center = pts.mean(axis=0)
            self.radius = float(np.max(np.linalg.norm(pts - self.center, axis=1)))
        else:
            self.center = np.zeros(2, dtype=float)
            self.radius = 0.0

    def __len__(self) -> int:
        return len(self.points_idx)




@dataclass
class GranularBallMedoid:
    points_idx: List[int]
    data: np.ndarray

    def __post_init__(self) -> None:
        self.points_idx = list(map(int, self.points_idx))
        self.data = np.asarray(self.data)
        self._center_radius()

    def _center_radius(self) -> None:
        if not self.points_idx:
            self.center = np.zeros(2, dtype=float)
            self.radius = 0.0
            return

        pts = self.data[self.points_idx]
        m = pts.shape[0]
        if m == 1:
            self.center = pts[0].copy()
            self.radius = 0.0
            return

        diff = pts[:, None, :] - pts[None, :, :]
        D = np.linalg.norm(diff, axis=2)
        medoid_local = int(np.argmin(D.sum(axis=1)))
        self.center = pts[medoid_local].copy()
        self.radius = float(np.max(D[:, medoid_local]))

    def __len__(self) -> int:
        return len(self.points_idx)



def balls_overlap(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float) -> bool:
    """球相交/重叠判定：dist(c1,c2) <= r1 + r2"""
    return float(np.linalg.norm(c1 - c2)) <= (float(r1) + float(r2))


def ball_contains(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float) -> bool:
    """包含判定：c1,r1 包含 c2,r2 <=> dist(c1,c2) + r2 <= r1"""
    return float(np.linalg.norm(c1 - c2)) + float(r2) <= float(r1)
