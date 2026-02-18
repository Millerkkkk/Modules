# coding: UTF-8
from __future__ import annotations

import math
from typing import Dict, List, Set, Optional

import numpy as np
from Clustering.methods.utils import feat_location, feat_time_window, feat_time_window_mid_width


# ---------------- clustering ----------------

class ImprovedKMeans:
    """
    Capacity-constrained (hard) k-means-like clustering.
    - Uses a feature_fn for coordinates (XY or TW mid/width, etc.).
    - Centers are mean of features.
    - Assignment is greedy with capacity feasibility.
    - k starts from ceil(total_demand / vehicle_capacity); can auto-increase if assignment fails.
    """

    def __init__(
        self,
        problem,
        customer_ids: List[int],
        *,
        feature: str = "location",              # "location" | "time_window" | "time_window_mid_width"
        vehicle_capacity: Optional[float] = None,
        
        tw_scale_mid: float = 1.0,              # used if feature="time_window_mid_width"
        tw_scale_width: float = 1.0,            # used if feature="time_window_mid_width"
        
        random_state: Optional[int] = None,
        max_iter: int = 100,

        tol: float = 1e-4,
        cache_features: bool = True,
    ) -> None:
        self.problem = problem
        self.customer_ids = list(customer_ids)

        self.vehicle_capacity = float(vehicle_capacity if vehicle_capacity is not None else problem.vehicle_capacity)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        self._rng = np.random.default_rng(random_state)

        # choose feature function
        if feature == "location":
            self._feat_fn = feat_location(problem)
            self._feat_dim = 2
        elif feature == "time_window_mid_width":
            self._feat_fn = feat_time_window_mid_width(problem, scale_mid=tw_scale_mid, scale_width=tw_scale_width)
            self._feat_dim = 2
        elif feature == "time_window":
            self._feat_fn = feat_time_window(problem)
            self._feat_dim = 2
        else:
            raise ValueError(
                f"Unknown feature={feature!r}, expected 'location', 'time_window', or 'time_window_mid_width'"
            )


        self.cache_features = bool(cache_features)
        self._feat_cache: Dict[int, np.ndarray] = {}
        if self.cache_features:
            for c in self.customer_ids:
                v = self._feat_fn(c).reshape(-1)
                if v.shape[0] != self._feat_dim:
                    raise ValueError(f"feature dim mismatch for {c}: {v.shape[0]} != {self._feat_dim}")
                self._feat_cache[c] = v.astype(float, copy=False)

        # results
        self.k: int = 0
        self.clusters: Dict[int, Set[int]] = {}
        self.cluster_capacities: Dict[int, float] = {}
        self.centers: Dict[int, np.ndarray] = {}

    # -------- internals --------

    def _feat(self, i: int) -> np.ndarray:
        if self.cache_features:
            return self._feat_cache[i]
        v = self._feat_fn(i).reshape(-1).astype(float, copy=False)
        if v.shape[0] != self._feat_dim:
            raise ValueError(f"feature dim mismatch for {i}: {v.shape[0]} != {self._feat_dim}")
        return v

    def _get_cluster_num(self) -> int:
        total_demand = sum(self.problem.demand[c] for c in self.customer_ids)
        return max(1, int(math.ceil(total_demand / self.vehicle_capacity)))

    def _init_centers(self) -> None:
        center_ids = self._rng.choice(self.customer_ids, size=self.k, replace=False)
        self.centers = {i: self._feat(int(cid)).copy() for i, cid in enumerate(center_ids)}

    def _assign_once(self) -> bool:
        unallocated = set(self.customer_ids)

        self.clusters = {i: set() for i in range(self.k)}
        self.cluster_capacities = {i: float(self.vehicle_capacity) for i in range(self.k)}

        centers_mat = np.vstack([self.centers[i] for i in range(self.k)])  # (k, D)

        while unallocated:
            c = max(unallocated, key=lambda idx: self.problem.demand[idx])
            d_c = self.problem.demand[c]
            v = self._feat(c)

            # squared euclidean distance for ranking
            diff = centers_mat - v
            d2 = np.einsum("ij,ij->i", diff, diff)
            sorted_idx = np.argsort(d2)

            for i in sorted_idx:
                i = int(i)
                if self.cluster_capacities[i] >= d_c:
                    self.clusters[i].add(c)
                    self.cluster_capacities[i] -= d_c
                    unallocated.remove(c)
                    break
            else:
                return False

        return True

    def _calculate_centers(self) -> Dict[int, np.ndarray]:
        new_centers: Dict[int, np.ndarray] = {}
        for i in range(self.k):
            if not self.clusters[i]:
                new_centers[i] = self.centers[i]
                continue
            mat = np.vstack([self._feat(c) for c in self.clusters[i]])
            new_centers[i] = mat.mean(axis=0)
        return new_centers

    def _max_center_shift(self, new_centers: Dict[int, np.ndarray]) -> float:
        return float(max(np.linalg.norm(new_centers[i] - self.centers[i]) for i in range(self.k)))

    @staticmethod
    def _clusters_equal(a: Dict[int, Set[int]], b: Dict[int, Set[int]]) -> bool:
        return a.keys() == b.keys() and all(a[i] == b[i] for i in a.keys())


    # -------- public --------

    def fit(self) -> Dict[int, Set[int]]:
        if not self.customer_ids:
            return {}

        # demand sanity (no split inside clustering)
        max_d = max(self.problem.demand[c] for c in self.customer_ids)
        if max_d > self.vehicle_capacity:
            raise ValueError(
                f"Found demand > capacity (max_demand={max_d}, capacity={self.vehicle_capacity}). "
                f"Either allow split or increase capacity."
            )

        base_k = self._get_cluster_num()

        # 固定 k = base_k，不再增加 k
        self.k = min(base_k, len(self.customer_ids))
        self._init_centers()

        prev_clusters: Optional[Dict[int, Set[int]]] = None

        for _ in range(self.max_iter):
            ok = self._assign_once()
            if not ok:
                raise RuntimeError(
                f"Failed to find feasible clustering with fixed k={self.k}. "
                f"Try a different random_state or adjust vehicle_capacity."
            )

            new_centers = self._calculate_centers()
            shift = self._max_center_shift(new_centers)

            if prev_clusters is not None and self._clusters_equal(prev_clusters, self.clusters):
                self.centers = new_centers
                return self.clusters

            self.centers = new_centers
            if shift < self.tol:
                return self.clusters

            prev_clusters = {i: set(s) for i, s in self.clusters.items()}

        return self.clusters
    
