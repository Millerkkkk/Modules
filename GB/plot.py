from __future__ import annotations

from typing import List, Optional, Sequence, Dict, Any
import numpy as np
import matplotlib.pyplot as plt


def plot_clusters(
    points: np.ndarray,
    clusters: List[List[int]],
    *,
    centers: Optional[np.ndarray] = None,
    radii: Optional[Sequence[float]] = None,
    title: str = "GB Clustering Result",
    show_index: bool = False,
):
    """
    通用聚类可视化函数

    参数
    ----
    points:
        ndarray (n,2)

    clusters:
        List[List[int]]
        每个 cluster 是点索引集合

    centers:
        ndarray (k,2)，可选
        每个 cluster 的中心点

    radii:
        list[float]，可选
        每个 cluster 的半径（用于画粒球圆）

    title:
        图标题

    show_index:
        是否在点旁标注索引（调试用）
    """
    pts = np.asarray(points, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be shape (n,2), got {pts.shape}")

    plt.figure(figsize=(8, 6))

    # --- 画 clusters ---
    for cid, idxs in enumerate(clusters):
        idxs = list(map(int, idxs))
        cluster_pts = pts[idxs]

        plt.scatter(
            cluster_pts[:, 0],
            cluster_pts[:, 1],
            s=25,
            label=f"Cluster {cid}",
        )

        # 可选标注索引
        if show_index:
            for i in idxs:
                plt.text(
                    pts[i, 0],
                    pts[i, 1],
                    str(i),
                    fontsize=8,
                )

    # --- 画 centers ---
    if centers is not None and len(centers) > 0:
        centers = np.asarray(centers, dtype=float)
        plt.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="x",
            s=80,
            linewidths=2,
            label="Centers",
        )

    # --- 画 radii 圆 ---
    if radii is not None and centers is not None:
        radii = list(map(float, radii))
        for c, r in zip(centers, radii):
            circle = plt.Circle(
                (c[0], c[1]),
                r,
                fill=False,
                linestyle="--",
                linewidth=1.2,
            )
            plt.gca().add_patch(circle)

    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_result(
    points: np.ndarray,
    result: tuple,
    *,
    title: str = "GB Result",
):
    """
    快捷接口：直接接收 cluster(...) 的返回结果

    用法：
      clusters, meta = cluster(points, return_meta=True)
      plot_result(points, (clusters, meta))

    或：
      clusters = cluster(points)
      plot_result(points, (clusters, {}))
    """
    clusters = result[0]
    meta: Dict[str, Any] = result[1] if len(result) > 1 else {}

    centers = meta.get("centers", None)
    radii = meta.get("radii", None)

    plot_clusters(
        points,
        clusters,
        centers=centers,
        radii=radii,
        title=title,
    )
