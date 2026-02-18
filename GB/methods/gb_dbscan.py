from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import distance
from pyflann import FLANN, set_distance_type

from ..gb import GranularBall, balls_overlap
from ..params import GBClusterResult


class GbDbscanCluster:
    """
    说明：
    - 通用化入口：输入 points (n,2)，输出分组索引
    - 流程：生成粒球 -> core/non-core -> core聚类 -> 分配非core -> 输出结果
    - 不输出噪声：最后兜底把 -1 分配到最近 core（或无 core 时全归 0）
    """
    def __init__(
            self, points: np.ndarray, 
            split_k: float = 0.3, 
            core_ratio: float = 0.92,
            ):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"points must be shape (n,2), got {pts.shape}")
        if len(pts) and np.any(~np.isfinite(pts)):
            raise ValueError("points contains NaN/Inf")
        if split_k < 0:
            raise ValueError("split_k must be >= 0")
        
        if not (0 < core_ratio <= 1):
            raise ValueError("core_ratio must be in (0, 1]")


        self.points = pts
        self.split_k = float(split_k)
        self.core_ratio = float(core_ratio)

        self.point_labels = np.full(len(self.points), -1, dtype=int)
        self.dist_matrix = distance.cdist(self.points, self.points)

        self.gbs: List[GranularBall] = []
        self.core_gbs: List[GranularBall] = []
        self.non_core_gbs: List[GranularBall] = []
        self.core_labels: List[int] = []

    def generate_gbs(self) -> None:
        n = len(self.points)
        if n == 0:
            self.gbs = []
            return

        set_distance_type("euclidean")
        flann = FLANN()

        K = int(np.ceil(np.sqrt(n) * self.split_k))
        K = min(max(K, 1), n)

        data = self.points.astype(np.float32, copy=False)

        nn, _ = flann.nn(
            data, data,
            num_neighbors=K,
            algorithm="kmeans",
            branching=32,
            iterations=7,
            checks=-1
        )
        nn = np.asarray(nn, dtype=int)

        visited = np.zeros(n, dtype=bool)
        gbs: List[GranularBall] = []

        for i in range(n):
            if visited[i]:
                continue
            local_points_idx = np.unique(nn[i]).astype(int).tolist()
            visited[local_points_idx] = True
            gbs.append(GranularBall(local_points_idx, self.points))

        self.gbs = gbs

    def partition_core_non_core(self) -> None:
        self.core_gbs, self.non_core_gbs = [], []
        if not self.gbs:
            return

        # density = radius（越小越密）
        densities = np.array([gb.radius for gb in self.gbs], dtype=float)

        k = int(len(self.gbs) * self.core_ratio)
        k = min(max(k, 1), len(self.gbs))
        threshold = np.sort(densities)[k - 1]

        for gb in self.gbs:
            if gb.radius <= threshold:
                self.core_gbs.append(gb)
            else:
                self.non_core_gbs.append(gb)

        self.density_threshold = float(threshold)

    def cluster_core_gbs(self) -> None:
        """core 球按“相交即连通”建图，连通分量作为簇标签（并查集）"""
        m = len(self.core_gbs)
        if m == 0:
            self.core_labels = []
            return

        parent = list(range(m))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(m):
            for j in range(i + 1, m):
                if balls_overlap(self.core_gbs[i].center, self.core_gbs[i].radius,
                                 self.core_gbs[j].center, self.core_gbs[j].radius):
                    union(i, j)

        root_to_lab: Dict[int, int] = {}
        labels = [-1] * m
        cur = 0
        for i in range(m):
            r = find(i)
            if r not in root_to_lab:
                root_to_lab[r] = cur
                cur += 1
            labels[i] = root_to_lab[r]

        self.core_labels = labels

        # 标记 core 球内点
        for gb_idx, gb in enumerate(self.core_gbs):
            lab = int(self.core_labels[gb_idx])
            for p in gb.points_idx:
                self.point_labels[p] = lab

    def assign_non_core_gbs(self) -> None:
        if not self.non_core_gbs:
            return

        if len(self.core_gbs) == 0:
            # 兜底：全部为 0
            self.point_labels[:] = 0
            return

        core_centers = np.array([gb.center for gb in self.core_gbs], dtype=float)
        core_labels = np.array(self.core_labels, dtype=int)

        for gb in self.non_core_gbs:
            known_points = [p for p in gb.points_idx if self.point_labels[p] != -1]
            unknown_points = [p for p in gb.points_idx if self.point_labels[p] == -1]

            if known_points:
                if not unknown_points:
                    continue

                dists = self.dist_matrix[np.ix_(unknown_points, known_points)]
                nearest = np.argmin(dists, axis=1)
                for t, p in enumerate(unknown_points):
                    nearest_known = known_points[int(nearest[t])]
                    self.point_labels[p] = int(self.point_labels[nearest_known])
                continue

            # 球内无已标记点：按球中心最近 core 球分配
            d = np.linalg.norm(core_centers - gb.center, axis=1)
            nearest_core = int(np.argmin(d))
            lab = int(core_labels[nearest_core])
            for p in gb.points_idx:
                self.point_labels[p] = lab

    def fit(self) -> Tuple[List[List[int]], np.ndarray, List[float]]:
        """
        返回:
          clusters: list[list[int]]（不含噪声；最终保证无 -1）
          centers:  (k,2)
          radii:    list[float]
        """
        self.point_labels[:] = -1
        self.gbs = []
        self.core_gbs = []
        self.non_core_gbs = []
        self.core_labels = []

        self.generate_gbs()
        self.partition_core_non_core()
        self.cluster_core_gbs()
        self.assign_non_core_gbs()

        # 兜底消除 -1（不要噪声）
        unassigned = np.where(self.point_labels == -1)[0]
        if len(unassigned) > 0:
            if len(self.core_gbs) == 0:
                self.point_labels[unassigned] = 0
            else:
                core_centers = np.array([gb.center for gb in self.core_gbs], dtype=float)
                core_labels = np.array(self.core_labels, dtype=int)
                d = distance.cdist(self.points[unassigned], core_centers)
                nearest = np.argmin(d, axis=1)
                self.point_labels[unassigned] = core_labels[nearest]

        clusters_dict: Dict[int, List[int]] = defaultdict(list)
        for i, lab in enumerate(self.point_labels):
            clusters_dict[int(lab)].append(i)

        clusters = [sorted(map(int, clusters_dict[k])) for k in sorted(clusters_dict.keys())]

        centers: List[np.ndarray] = []
        radii: List[float] = []
        for cl in clusters:
            pts = self.points[cl]
            c = pts.mean(axis=0)
            r = float(np.max(np.linalg.norm(pts - c, axis=1))) if len(cl) else 0.0
            centers.append(c)
            radii.append(r)

        centers_arr = np.vstack(centers) if centers else np.zeros((0, 2), dtype=float)
        return clusters, centers_arr, radii


def cluster_gb_dbscan(
    points: np.ndarray,
    *,
    split_k: float = 0.3,
    core_ratio: float = 0.92,
) -> GBClusterResult:
    """
    对外 wrapper：
      输入 points (n,2)
      输出 clusters: list[list[int]]，索引 0..n-1
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be shape (n,2), got {pts.shape}")
    if len(pts) and np.any(~np.isfinite(pts)):
        raise ValueError("points contains NaN/Inf")

    model = GbDbscanCluster(pts, split_k=split_k, core_ratio=core_ratio)
    clusters, centers, radii = model.fit()

    return GBClusterResult(
        gbs=clusters,
        centers=centers,
        radii=radii,
    )
