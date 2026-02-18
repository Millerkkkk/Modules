from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Union, Optional, Tuple, Iterable, Any

ClustersLike = Union[List[List[int]], Dict[int, Set[int]]]


def _iter_clusters(clusters: ClustersLike) -> Iterable[Tuple[int, List[int]]]:
    if isinstance(clusters, dict):
        for cid, members in clusters.items():
            yield int(cid), [int(x) for x in members]
    else:
        for cid, members in enumerate(clusters):
            yield int(cid), [int(x) for x in members]


def _feat(problem, cid: int, feature: str,
          tw_scale_mid: float, tw_scale_width: float) -> np.ndarray:

    if feature == "location":
        return np.array([problem.x[cid], problem.y[cid]], dtype=float)

    elif feature == "time_window":
        return np.array([problem.ready_time[cid], problem.due_time[cid]], dtype=float)

    elif feature == "time_window_mid_width":
        e, l = problem.ready_time[cid], problem.due_time[cid]
        # 建议：防御性检查，避免 NaN/inf 污染
        if not (np.isfinite(e) and np.isfinite(l)):
            return np.array([np.nan, np.nan], dtype=float)
        mid = 0.5 * (e + l)
        width = l - e
        return np.array([mid * tw_scale_mid, width * tw_scale_width], dtype=float)

    else:
        raise ValueError("feature must be 'location', 'time_window', or 'time_window_mid_width'")


def plot_clusters(
    problem,
    clusters: ClustersLike,
    *,
    feature: str = "location",
    tw_scale_mid: float = 1.0,
    tw_scale_width: float = 1.0,
    centers: Optional[Union[np.ndarray, Dict[int, np.ndarray]]] = None,
    title: Optional[str] = None,
    annotate: bool = False,
    figsize: Tuple[int, int] = (8, 6),
):
    plt.figure(figsize=figsize)

    for cid, members in _iter_clusters(clusters):
        if not members:
            continue

        pts = np.stack(
            [_feat(problem, c, feature, tw_scale_mid, tw_scale_width) for c in members],
            axis=0
        )

        # 过滤掉 nan 点，避免 matplotlib warning
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]
        if pts.size == 0:
            continue

        plt.scatter(pts[:, 0], pts[:, 1], s=25, label=f"Cluster {cid}")

        if annotate:
            for c, (x, y) in zip([m for m, ok in zip(members, mask) if ok], pts):
                plt.text(x, y, str(c), fontsize=8)

    if centers is not None:
        if isinstance(centers, dict):
            centers_arr = np.vstack([centers[i] for i in sorted(centers.keys())])
        else:
            centers_arr = np.asarray(centers, dtype=float)

        plt.scatter(
            centers_arr[:, 0],
            centers_arr[:, 1],
            marker="X",
            s=160,
            edgecolors="k",
            linewidths=1.0,
            label="Centers"
        )

    if title is None:
        if feature == "location":
            title = "Clusters (location)"
            xlabel, ylabel = "x", "y"
        elif feature == "time_window":
            title = "Clusters (ready, due)"
            xlabel, ylabel = "ready_time", "due_time"
        else:
            title = "Clusters (mid, width)"
            xlabel, ylabel = "mid (scaled)", "width (scaled)"
    else:
        xlabel, ylabel = "dim1", "dim2"

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_result(
    problem,            # 注意：这里应当传 problem，而不是 points
    result: tuple,
    *,
    title: str = "GB Result",
):
    clusters = result[0]
    meta: Dict[str, Any] = result[1] if len(result) > 1 else {}

    centers = meta.get("centers", None)

    plot_clusters(
        problem,
        clusters,
        centers=centers,
        title=title,
    )
