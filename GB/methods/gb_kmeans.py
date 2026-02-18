from __future__ import annotations

from typing import Dict, List, Optional
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from ..gb import GranularBall
from ..params import GBClusterResult


class KMeansCluster:
    def __init__(
        self,
        points: np.ndarray,
        split_k: float = 1.0,              # 固定叫 split_k
        random_state: int | None = None,
    ):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"points must be shape (n,2), got {pts.shape}")
        if len(pts) and np.any(~np.isfinite(pts)):
            raise ValueError("points contains NaN/Inf")
        if split_k < 0:
            raise ValueError("split_k must be >= 0")

        self.points = pts
        self.split_k = float(split_k)
        if random_state is None:
            random_state = np.random.randint(0, 10**9)
        self.random_state = random_state

        # threshold 固定逻辑：split_k * sqrt(n)
        self.threshold = self.split_k * float(np.sqrt(len(self.points)))

        self.gbs: List[GranularBall] = []

    def _split_gb(self, gb: GranularBall) -> List[List[int]]:
        """
        用 KMeans(2) 分裂一个球，返回两个子球的 points_idx 列表
        """
        gb_idx = gb.points_idx
        if len(gb_idx) <= 2:
            return [gb_idx]

        cur_idx = np.asarray(gb_idx, dtype=int)
        pts = self.points[cur_idx]  # (m,2)

        km = KMeans(n_clusters=2, random_state=self.random_state, n_init="auto")
        # km = KMeans(n_clusters=2)
        labels = km.fit_predict(pts)

        left = cur_idx[labels == 0].tolist()
        right = cur_idx[labels == 1].tolist()

        # # 防止退化
        # if not left or not right:
        #     return [gb_idx]

        return [left, right]

    def generate_gbs(self):
        """
        while True 的批量 split：
        - 初始只有一个大球
        - 每轮：对所有球，len < threshold 不拆，len >= threshold 拆
        - 球数量不变则停止
        结果写入 self.gbs
        """
        n = len(self.points)
        self.gbs = []
        if n == 0:
            return
        # 初始一个大球
        gbs_list: List[List[int]] = [list(range(n))]

        while True:
            before = len(gbs_list)
            new_list: List[List[int]] = []

            for gb_idx in gbs_list:
                if len(gb_idx) < self.threshold:
                    new_list.append(gb_idx)
                else:
                    # 注意：这里用临时 GranularBall 只是为了复用 _split_gb
                    tmp_gb = GranularBall(gb_idx, self.points)
                    new_list.extend(self._split_gb(tmp_gb))

            gbs_list = new_list
            after = len(gbs_list)
            if after == before:
                break

        # ✅ 最终统一构造成 GranularBall 对象
        self.gbs = [GranularBall(sorted(set(idx)), self.points) for idx in gbs_list]

    def merge_single_point_gb(self) -> None:
        """
        把单点球并入最近的非单点球（按点到球中心距离）
        静态：使用“合并前”的 centers
        """
        if not self.gbs:
            return

        singles = [gb for gb in self.gbs if len(gb) == 1]
        others = [gb for gb in self.gbs if len(gb) > 1]
        if not singles or not others:
            return

        other_centers = np.vstack([gb.center for gb in others])  # (m,2)
        new_idxs = [list(gb.points_idx) for gb in others]

        for sg in singles:
            p = int(sg.points_idx[0])
            xy = self.points[p]
            j = int(np.argmin(np.linalg.norm(other_centers - xy, axis=1)))
            new_idxs[j].append(p)

        self.gbs = [GranularBall(sorted(set(idx)), self.points) for idx in new_idxs]
        
    def _merge_contained_gbs(self, merged_gbs):
        """
        merged_gbs: Dict[int, GranularBall]
        return: Dict[int, GranularBall]  (after containment merge)
        """
        # label -> (center, radius)
        centers_radii = {lab: (gb.center, gb.radius) for lab, gb in merged_gbs.items()}

        containment_relations = {}
        for lab, (c, r) in centers_radii.items():
            for other_lab, (oc, or_) in centers_radii.items():
                if lab == other_lab:
                    continue
                d = np.linalg.norm(np.asarray(c) - np.asarray(oc))
                if d + r <= or_:  # lab inside other_lab
                    containment_relations.setdefault(other_lab, []).append(lab)

        # 合并 contained -> encloser
        new_merged = {}
        for encloser, contained in containment_relations.items():
            if encloser not in new_merged:
                new_merged[encloser] = list(merged_gbs[encloser].points_idx)
            for lab in contained:
                new_merged[encloser].extend(merged_gbs[lab].points_idx)

        # 加入未被包含的球
        contained_all = set(sum(containment_relations.values(), []))
        for lab, gb in merged_gbs.items():
            if lab in contained_all:
                continue
            if lab not in new_merged:
                new_merged[lab] = list(gb.points_idx)

        # 重建为 GranularBall（去重+排序）
        out = {lab: GranularBall(sorted(set(idx)), self.points) for lab, idx in new_merged.items()}
        return out

    def merge_gbs(self):
        """
        Combining overlapping or contained balls, using self.gbs only.
        overlap rule: dist(ci,cj) <= ri + rj
        then containment merge
        """
        m = len(self.gbs)
        if m <= 1:
            return self.gbs

        centers = np.vstack([gb.center for gb in self.gbs]).astype(float)   # (m,2)
        radii = np.asarray([gb.radius for gb in self.gbs], dtype=float)     # (m,)

        unvisited = list(range(m))
        comp = [-1] * m
        k = -1

        while unvisited:
            p = unvisited[0]
            unvisited.remove(p)

            neighbors = []
            for i in range(m):
                if i == p:
                    continue
                d = np.linalg.norm(centers[i] - centers[p])
                if d <= (radii[i] + radii[p]):
                    neighbors.append(i)

            k += 1
            comp[p] = k

            for pi in neighbors:
                if pi in unvisited:
                    unvisited.remove(pi)

                    neighbors_pi = []
                    for j in range(m):
                        if j == pi:
                            continue
                        d_pi = np.linalg.norm(centers[j] - centers[pi])
                        if d_pi <= (radii[j] + radii[pi]):
                            neighbors_pi.append(j)

                    for t in neighbors_pi:
                        if t not in neighbors:
                            neighbors.append(t)

                if comp[pi] == -1:
                    comp[pi] = k

        # overlap 合并：label -> GranularBall（先收集索引）
        merged_idxs = {}
        for i, lab in enumerate(comp):
            merged_idxs.setdefault(int(lab), []).extend(self.gbs[i].points_idx)

        merged_gbs = {lab: GranularBall(sorted(set(idx)), self.points) for lab, idx in merged_idxs.items()}

        # contain 合并（不依赖 _gb_center_radius）
        merged_gbs = self._merge_contained_gbs(merged_gbs)

        # 写回 self.gbs（按 label 排序稳定）
        self.gbs = [merged_gbs[lab] for lab in sorted(merged_gbs.keys())]
        return self.gbs
    


    def fit(self, merge: bool = True):
        """
        pipeline:
          1) generate_gbs
          2) merge single-point
          3) optional merge overlap + contain（你若有就加）
          4) return clusters, centers, radii
        """
        self.generate_gbs()
        self.merge_single_point_gb()

        # for i, gb in enumerate(self.gbs):
        #     print(f"GB {i}: points_idx = {gb.points_idx}")

        if merge:
            self.merge_gbs()

        clusters = [sorted(gb.points_idx) for gb in self.gbs]
        centers_arr = np.vstack([gb.center for gb in self.gbs]) if self.gbs else np.zeros((0, 2), dtype=float)
        radii_arr = np.asarray([gb.radius for gb in self.gbs], dtype=float)
        return clusters, centers_arr, radii_arr



def cluster_gb_kmeans(
    points: np.ndarray,
    *,
    split_k: float = 1.0,
    merge: bool = False,
    random_state: int = 0,
) -> GBClusterResult:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be shape (n,2), got {pts.shape}")
    if len(pts) and np.any(~np.isfinite(pts)):
        raise ValueError("points contains NaN/Inf")

    model = KMeansCluster(pts, split_k=split_k, random_state=random_state)
    clusters, centers, radii = model.fit(merge=merge)

    return GBClusterResult(gbs=clusters, centers=centers, radii=radii)
