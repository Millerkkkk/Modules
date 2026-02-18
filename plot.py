from typing import Dict, List, Optional



from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt






def plot_routes(problem, routes,
                title="EVRP Routes", annotate=True, save_path=None):
    """可视化 EVRP 路线（终版）。
    兼容两种输入：
      1) dict[int, list[int]]          # 每个 key 一条完整路线（已 merge）
      2) list[list[int]]               # 多条路线

    Args:
        problem: EVRPProblem（需有 x,y, depots, css, customers）
        routes:  dict[int, list[int]] 或 list[list[int]]
    """

    # ---- normalize routes -> list[list[int]] ----
    if routes is None:
        route_list = []
    elif isinstance(routes, dict):
        # 保持 key 顺序稳定（按 key 排序），便于对齐颜色/复现实验
        route_list = [routes[k] for k in sorted(routes.keys())]
    else:
        route_list = list(routes)

    # ---- coords ----
    coords = np.column_stack((np.asarray(problem.x, dtype=float),
                              np.asarray(problem.y, dtype=float)))
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"problem.x/y must form (N,2) coords, got {coords.shape}")
    n = coords.shape[0]

    # ---- indices ----
    dep_idx  = np.asarray(getattr(problem, "depots", []), dtype=int)
    cs_idx   = np.asarray(getattr(problem, "css", []), dtype=int)
    cust_idx = np.asarray(getattr(problem, "customers", []), dtype=int)

    def _check_idx(name, idx):
        if idx.size == 0:
            return
        if np.any(idx < 0) or np.any(idx >= n):
            bad = idx[(idx < 0) | (idx >= n)]
            raise IndexError(f"{name} contains out-of-range indices: {bad.tolist()} (valid: 0..{n-1})")

    _check_idx("problem.depots", dep_idx)
    _check_idx("problem.css", cs_idx)
    _check_idx("problem.customers", cust_idx)

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(8, 7))

    # nodes
    dep = ax.scatter(coords[dep_idx, 0],  coords[dep_idx, 1],  s=80, marker='s', c='#D02C1B', label='Depot', zorder=3)
    cs  = ax.scatter(coords[cs_idx, 0],   coords[cs_idx, 1],   s=70, marker='^', c='#25A61F', label='CS', zorder=3)
    cus = ax.scatter(coords[cust_idx, 0], coords[cust_idx, 1], s=35, marker='o', c='#3179B5', label='Customer', zorder=3)

    # routes
    seen_annot = set()
    for r in route_list:
        if r is None:
            continue
        rr = np.asarray(r, dtype=int)
        if rr.size < 2:
            continue
        if np.any(rr < 0) or np.any(rr >= n):
            # 输入存在非法 idx：跳过该条路线（如需严格可改为 raise）
            continue

        pts = coords[rr]
        ax.plot(pts[:, 0], pts[:, 1], linewidth=2, alpha=0.9, zorder=2)

        if annotate:
            for nid in rr:
                nid_int = int(nid)
                if nid_int in seen_annot:
                    continue
                x, y = coords[nid_int]
                ax.text(x + 0.5, y + 0.8, f"{nid_int}",
                        fontsize=9, ha='center', va='center', color="black", zorder=4)
                seen_annot.add(nid_int)

    # styling
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    ax.grid(True, alpha=0.3)

    # bounds
    xmn, xmx = np.nanmin(coords[:, 0]), np.nanmax(coords[:, 0])
    ymn, ymx = np.nanmin(coords[:, 1]), np.nanmax(coords[:, 1])
    dx = max((xmx - xmn) * 0.05, 1e-6)
    dy = max((ymx - ymn) * 0.05, 1e-6)
    ax.set_xlim(xmn - dx, xmx + dx)
    ax.set_ylim(ymn - dy, ymx + dy)

    # legend (nodes only)
    ax.legend(handles=[cus, cs, dep], loc='best')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax



def plot_clusters_gbs(problem,
                          clusters,
                          title="Clusters + Granular Balls",
                          show_nodes=True,
                          show_circles=True,
                          annotate_gb=False,
                          annotate_points=False,
                          alpha_points=0.85,
                          alpha_circle=0.55,
                          save_path=None):
    """
    在一张图上画：
      - 所有簇（不同簇不同颜色）
      - 每簇内多个颗粒球GB（画圆边界）
      - 球内点（散点）
    clusters: dict[int, list[list[int]]]
    problem: 需要有 x, y，以及可选 depots/css/customers
    """

    # ---- coords ----
    coords = np.column_stack((np.asarray(problem.x, dtype=float),
                              np.asarray(problem.y, dtype=float)))
    n = coords.shape[0]

    # ---- indices (optional) ----
    dep_idx  = np.asarray(getattr(problem, "depots", []), dtype=int)
    cs_idx   = np.asarray(getattr(problem, "css", []), dtype=int)
    cust_idx = np.asarray(getattr(problem, "customers", []), dtype=int)

    fig, ax = plt.subplots(figsize=(9, 8))

    # ---- base nodes ----
    handles = []
    if show_nodes:
        if dep_idx.size:
            h = ax.scatter(coords[dep_idx, 0], coords[dep_idx, 1],
                           s=50, marker='s', label='Depot', zorder=4)
            handles.append(h)
        if cs_idx.size:
            h = ax.scatter(coords[cs_idx, 0], coords[cs_idx, 1],
                           s=40, marker='^', label='CS', zorder=4)
            handles.append(h)
        if cust_idx.size:
            h = ax.scatter(coords[cust_idx, 0], coords[cust_idx, 1],
                           s=25, marker='o', alpha=0.25, label='Customer (all)', zorder=1)
            handles.append(h)

    # ---- colors per cluster ----
    cluster_keys = sorted(clusters.keys())
    cmap = cm.get_cmap('tab10', max(len(cluster_keys), 1))

    # ---- plot each cluster ----
    for ci, cid in enumerate(cluster_keys):
        color = cmap(ci)

        gbs = clusters[cid]  # list[list[int]]
        # 聚合该簇所有点（方便一次性画背景）
        cluster_points = sorted({p for gb in gbs for p in gb if 0 <= p < n})
        if not cluster_points:
            continue

        pts = coords[cluster_points]
        # 簇点（颜色区分簇）
        ax.scatter(pts[:, 0], pts[:, 1],
                   s=25, alpha=alpha_points, zorder=2, label=f"Cluster {cid}")

        # 每个GB：画圆（可选）+ 可选标GB编号
        for gi, gb in enumerate(gbs):
            gb = [p for p in gb if 0 <= p < n]
            if len(gb) < 2:
                # 单点GB：可只画一个更大的点
                if len(gb) == 1:
                    p = coords[gb[0]]
                    ax.scatter([p[0]], [p[1]], s=80, alpha=0.95, zorder=3)
                    if annotate_points:
                        ax.text(p[0]+0.3, p[1]+0.3, str(int(gb[0])), fontsize=8, zorder=5)
                continue

            P = coords[np.asarray(gb, int)]
            center = P.mean(axis=0)
            radius = np.max(np.linalg.norm(P - center, axis=1))

            if show_circles:
                circ = plt.Circle((center[0], center[1]), radius,
                                  fill=False, linewidth=1,
                                  alpha=alpha_circle, zorder=3)
                ax.add_patch(circ)

            if annotate_gb:
                ax.text(center[0], center[1], f"{cid}-{gi}",
                        fontsize=9, ha='center', va='center', zorder=5)

            if annotate_points:
                for nid in gb:
                    x, y = coords[int(nid)]
                    ax.text(x+0.3, y+0.3, str(int(nid)), fontsize=7, zorder=5)

    # ---- styling ----
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    ax.grid(True, alpha=0.3)

    # bounds
    xmn, xmx = np.nanmin(coords[:, 0]), np.nanmax(coords[:, 0])
    ymn, ymx = np.nanmin(coords[:, 1]), np.nanmax(coords[:, 1])
    dx = max((xmx - xmn) * 0.05, 1e-6)
    dy = max((ymx - ymn) * 0.05, 1e-6)
    ax.set_xlim(xmn - dx, xmx + dx)
    ax.set_ylim(ymn - dy, ymx + dy)

    # legend
    ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.show()
    return fig, ax






def plot_clusters_gbs_with_gb_routes(
    problem,
    clusters: Dict[int, List[List[int]]],
    gb_routes: Optional[List[List[int]]] = None,   # 形如 [[gb1,gb2,...], [...], ...] 每簇一条GB路线
    title: str = "Clusters + Granular Balls + GB Routes",
    show_nodes: bool = True,
    show_circles: bool = True,
    annotate_gb: bool = False,
    annotate_points: bool = False,
    alpha_points: float = 0.85,
    alpha_circle: float = 0.55,
    route_linewidth: float = 2.0,
    route_alpha: float = 0.9,
    save_path: Optional[str] = None,
):
    """
    在一张图上画：
      - 所有簇（不同簇不同颜色）
      - 每簇内多个颗粒球 GB（画圆边界）
      - 球内点（散点）
      - 以及：每个簇内“GB之间的路线”（把GB当点，用GB center连线）

    参数：
      clusters: dict[int, list[list[int]]]
        key=cluster_id, value=该簇的GB列表，每个GB是点索引list
      gb_routes: list[list[int]]
        与 cluster_keys(sorted(clusters.keys())) 对齐：第 i 个簇的一条GB路线（GB编号序列）
        例如：gb_routes[i] = [2,1,0,3] 表示该簇的第2个GB->第1个GB->第0个GB->第3个GB
    """

    # ---- coords ----
    coords = np.column_stack((
        np.asarray(problem.x, dtype=float),
        np.asarray(problem.y, dtype=float)
    ))
    n = coords.shape[0]

    # ---- indices (optional) ----
    dep_idx  = np.asarray(getattr(problem, "depots", []), dtype=int)
    cs_idx   = np.asarray(getattr(problem, "css", []), dtype=int)
    cust_idx = np.asarray(getattr(problem, "customers", []), dtype=int)

    fig, ax = plt.subplots(figsize=(9, 8))

    # ---- base nodes ----
    if show_nodes:
        if dep_idx.size:
            ax.scatter(coords[dep_idx, 0], coords[dep_idx, 1],
                       s=50, marker='s', label='Depot', zorder=4)
        if cs_idx.size:
            ax.scatter(coords[cs_idx, 0], coords[cs_idx, 1],
                       s=40, marker='^', label='CS', zorder=4)
        if cust_idx.size:
            ax.scatter(coords[cust_idx, 0], coords[cust_idx, 1],
                       s=25, marker='o', alpha=0.25, label='Customer (all)', zorder=1)

    # ---- colors per cluster ----
    cluster_keys = sorted(clusters.keys())
    cmap = cm.get_cmap('tab10', max(len(cluster_keys), 1))

    # ---- cache GB centers per cluster for routes ----
    # gb_centers_by_cluster[cluster_id] = list of centers aligned with clusters[cluster_id] gbs order
    gb_centers_by_cluster: Dict[int, List[Optional[np.ndarray]]] = {}

    # ---- plot each cluster: points + circles ----
    for ci, cid in enumerate(cluster_keys):
        color = cmap(ci)

        gbs = clusters[cid]  # list[list[int]]

        # 聚合该簇所有点
        cluster_points = sorted({p for gb in gbs for p in gb if 0 <= p < n})
        if cluster_points:
            pts = coords[cluster_points]
            ax.scatter(pts[:, 0], pts[:, 1],
                       s=25, alpha=alpha_points, zorder=2, label=f"Cluster {cid}")

        # 逐GB画圆 + 计算并缓存center
        centers_list: List[Optional[np.ndarray]] = []
        for gi, gb in enumerate(gbs):
            gb = [p for p in gb if 0 <= p < n]

            if len(gb) == 0:
                centers_list.append(None)
                continue

            P = coords[np.asarray(gb, dtype=int)]
            center = P.mean(axis=0)
            centers_list.append(center)

            if len(gb) < 2:
                # 单点GB：画大点（可选）
                pxy = coords[int(gb[0])]
                ax.scatter([pxy[0]], [pxy[1]], s=80, alpha=0.95, zorder=3)
                if annotate_points:
                    ax.text(pxy[0] + 0.3, pxy[1] + 0.3, str(int(gb[0])),
                            fontsize=8, zorder=5)
                continue

            radius = float(np.max(np.linalg.norm(P - center, axis=1)))

            if show_circles:
                circ = plt.Circle((center[0], center[1]), radius,
                                  fill=False, linewidth=1,
                                  alpha=alpha_circle, zorder=3)
                ax.add_patch(circ)

            if annotate_gb:
                ax.text(center[0], center[1], f"{cid}-{gi}",
                        fontsize=9, ha='center', va='center', zorder=5)

            if annotate_points:
                for nid in gb:
                    x, y = coords[int(nid)]
                    ax.text(x + 0.3, y + 0.3, str(int(nid)),
                            fontsize=7, zorder=5)

        gb_centers_by_cluster[cid] = centers_list

    # ---- draw GB routes (GB as nodes) ----
    if gb_routes is not None:
        # 与 cluster_keys 顺序对齐：gb_routes[i] 对应 cluster_keys[i]
        for ci, cid in enumerate(cluster_keys):
            if ci >= len(gb_routes):
                continue
            route = gb_routes[ci]
            if not route:
                continue

            centers_list = gb_centers_by_cluster.get(cid, [])
            if not centers_list:
                continue

            route_centers = []
            for gb_idx in route:
                if 0 <= gb_idx < len(centers_list):
                    c = centers_list[gb_idx]
                    if c is not None:
                        route_centers.append(c)

            if len(route_centers) >= 2:
                route_centers = np.vstack(route_centers)
                ax.plot(
                    route_centers[:, 0],
                    route_centers[:, 1],
                    linewidth=route_linewidth,
                    alpha=route_alpha,
                    zorder=6,
                    color=cmap(ci),   # 路线颜色与簇颜色一致
                )

    # ---- styling ----
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    ax.grid(True, alpha=0.3)

    xmn, xmx = np.nanmin(coords[:, 0]), np.nanmax(coords[:, 0])
    ymn, ymx = np.nanmin(coords[:, 1]), np.nanmax(coords[:, 1])
    dx = max((xmx - xmn) * 0.05, 1e-6)
    dy = max((ymx - ymn) * 0.05, 1e-6)
    ax.set_xlim(xmn - dx, xmx + dx)
    ax.set_ylim(ymn - dy, ymx + dy)

    ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.show()
    return fig, ax


