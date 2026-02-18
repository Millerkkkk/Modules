# coding: UTF-8
from __future__ import annotations

import math
from typing import Dict, List, Set, Optional

import numpy as np
from Clustering.methods.utils import feat_location, feat_time_window, feat_time_window_mid_width


class ImprovedKMedoids:
    """
    Capacity-constrained k-medoids clustering.

    Supported features:
    - "location"              -> use spatial distance matrix
    - "time_window"           -> use (ready, due) distance
    - "time_window_mid_width" -> use (mid, width) distance

    Output:
    - clusters: Dict[int, Set[int]]   (customer_id sets)
    """

    def __init__(
        self,
        problem,
        customer_ids: List[int],
        *,
        feature: str = "location",
        vehicle_capacity: Optional[float] = None,
        tw_scale_mid: float = 1.0,
        tw_scale_width: float = 1.0,
        random_state: Optional[int] = None,
        max_iter: int = 50,
        tol: float = 1e-4,
        cache_features: bool = True,
    ) -> None:

        self.problem = problem
        self.customer_ids = list(customer_ids)

        self.vehicle_capacity = float(
            vehicle_capacity if vehicle_capacity is not None else problem.vehicle_capacity
        )

        self.max_iter = int(max_iter)
        self.tol = float(tol)

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
        self.medoids: Dict[int, int] = {}              # cluster_id -> customer_id
        self.centers: Dict[int, np.ndarray] = {}       # cluster_id -> medoid feature vector

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
        # initialize medoids from customers; centers are their feature vectors
        medoid_ids = self._rng.choice(self.customer_ids, size=self.k, replace=False)
        self.medoids = {i: int(cid) for i, cid in enumerate(medoid_ids)}
        self.centers = {i: self._feat(int(cid)).copy() for i, cid in enumerate(medoid_ids)}

    def _assign_once(self) -> bool:
        unallocated = set(self.customer_ids)

        self.clusters = {i: set() for i in range(self.k)}
        self.cluster_capacities = {i: float(self.vehicle_capacity) for i in range(self.k)}

        centers_mat = np.vstack([self.centers[i] for i in range(self.k)])  # (k, D)

        while unallocated:
            c = max(unallocated, key=lambda idx: self.problem.demand[idx])
            d_c = self.problem.demand[c]
            v = self._feat(c)

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
        """
        k-medoids update:
        For each cluster, choose medoid = member minimizing sum of distances to other members.
        centers[i] becomes feat(medoid).
        """
        new_centers: Dict[int, np.ndarray] = {}
        new_medoids: Dict[int, int] = {}

        for i in range(self.k):
            members = sorted(self.clusters[i])
            if not members:
                # keep previous
                new_medoids[i] = self.medoids[i]
                new_centers[i] = self.centers[i]
                continue

            # build feature matrix for members
            F = np.vstack([self._feat(c) for c in members])  # (s, D)

            # pairwise distances (s, s)
            diff = F[:, None, :] - F[None, :, :]
            dist = np.linalg.norm(diff, axis=2)

            # medoid is argmin row-sum
            best_pos = int(np.argmin(dist.sum(axis=1)))
            med = int(members[best_pos])

            new_medoids[i] = med
            new_centers[i] = self._feat(med).copy()

        self.medoids = new_medoids
        return new_centers

    def _max_center_shift(self, new_centers: Dict[int, np.ndarray]) -> float:
        return float(max(np.linalg.norm(new_centers[i] - self.centers[i]) for i in range(self.k)))

    @staticmethod
    def _clusters_equal(a: Dict[int, Set[int]], b: Dict[int, Set[int]]) -> bool:
        return a.keys() == b.keys() and all(a[i] == b[i] for i in a.keys())

    def fit(self) -> Dict[int, Set[int]]:
        if not self.customer_ids:
            return {}

        # demand sanity
        max_d = max(self.problem.demand[c] for c in self.customer_ids)
        if max_d > self.vehicle_capacity:
            raise ValueError(
                f"Found demand > capacity (max_demand={max_d}, capacity={self.vehicle_capacity}). "
                f"Either allow split or increase capacity."
            )

        base_k = self._get_cluster_num()
        self.k = min(base_k, len(self.customer_ids))
        self._init_centers()

        prev_clusters: Optional[Dict[int, Set[int]]] = None
        prev_medoids: Optional[Dict[int, int]] = None

        for _ in range(self.max_iter):
            ok = self._assign_once()
            if not ok:
                raise RuntimeError(
                    f"Failed to find feasible clustering with fixed k={self.k}. "
                    f"Try a different random_state or adjust vehicle_capacity."
                )

            new_centers = self._calculate_centers()
            shift = self._max_center_shift(new_centers)

            # stronger stop: medoids unchanged
            if prev_medoids is not None and self.medoids == prev_medoids:
                self.centers = new_centers
                return self.clusters

            if prev_clusters is not None and self._clusters_equal(prev_clusters, self.clusters):
                self.centers = new_centers
                return self.clusters

            self.centers = new_centers
            if shift < self.tol:
                return self.clusters

            prev_clusters = {i: set(s) for i, s in self.clusters.items()}
            prev_medoids = dict(self.medoids)

        return self.clusters
    
