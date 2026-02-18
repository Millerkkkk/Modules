from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence, Tuple, overload
import numpy as np

from .params import GBClusterResult
from .methods.gb_dbscan import cluster_gb_dbscan
from .methods.gb_kmeans import cluster_gb_kmeans
from .methods.gb_kmedoids import cluster_gb_kmedoids   # ✅ 新增

from .methods.gbKmeans_original import gbs_generator   # ✅ 新增

GBMethod = Literal["gb_dbscan", "gb_kmeans", "gb_kmedoids", "gbs_generator"]

    

def _as_points(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be shape (n, 2), got {pts.shape}")
    if len(pts) == 0:
        return pts
    if np.any(~np.isfinite(pts)):
        raise ValueError("points contains NaN/Inf")
    return pts



def gb_clustering(
    points: Sequence[Sequence[float]] | np.ndarray,
    *,
    method: GBMethod = "gb_dbscan",
    **kwargs: Any,
) -> GBClusterResult:
    pts = _as_points(points)

    if method == "gb_dbscan":
        return cluster_gb_dbscan(pts, **kwargs)
    if method == "gb_kmeans":
        return cluster_gb_kmeans(pts, **kwargs)
    if method == "gb_kmedoids":
        return cluster_gb_kmedoids(pts, **kwargs)
    if method == "gbs_generator":  # ✅ 新增分支
        return gbs_generator(pts, **kwargs)

    raise ValueError(f"Unknown method: {method}")

