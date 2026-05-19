import copy
import inspect
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Callable

RoutesByClusters = Dict[int, List[int]]  # cid -> flat customer route


class NeighborhoodNoGB:
    """
    Neighborhood operators for NoGB / flat-route solutions.

    Solution representation:
        routes_by_clusters = {
            cid: [customer_1, customer_2, ..., customer_n]
        }

    - Input/Output: RoutesByClusters -> RoutesByClusters
    - Each operator performs ONE move (not enumeration).
    - 'cid' semantics:
        cid=None : pick a feasible cluster randomly (if needed)
        cid=int  : apply move only within that cluster
    """

    def __init__(
        self,
        problem,
        alpha_boundary: float = 0.5,
        removal_ratio: float = 0.3,
        rng: Optional[np.random.Generator] = None,
        evaluator: Optional[Callable[[List[int]], Any]] = None,
    ):
        self.problem = problem
        self.total_distance_matrix = self.problem.distance_matrix
        self.cap = self.problem.vehicle_capacity
        self.depot_ids = self.problem.depots

        self.removal_ratio = float(removal_ratio)
        self.alpha_boundary = float(alpha_boundary)
        self.rng = rng if rng is not None else np.random.default_rng()

        # optional lightweight evaluator for insertion cost, if needed
        self.evaluator = evaluator

    # ============================================================
    # Helpers
    # ============================================================
    @staticmethod
    def deep_copy_routes(routes_by_clusters: RoutesByClusters) -> RoutesByClusters:
        return copy.deepcopy(routes_by_clusters)

    @staticmethod
    def _proximity(idx: int, length: int) -> float:
        """Normalized boundary proximity: closer to route ends -> larger."""
        if length <= 1:
            return 0.0
        return 1.0 - min(idx, length - 1 - idx) / (length - 1)

    def _pick_cluster(self, cands: List[int]) -> Optional[int]:
        if not cands:
            return None
        return int(self.rng.choice(cands))

    def _route_load(self, route: List[int]) -> float:
        return sum(self.problem.demand[n] for n in route)

    def _get_nearest_depot(self, customer: int) -> int:
        """
        Return the nearest depot id to the given customer.
        """
        dist_to_depots = self.total_distance_matrix[self.depot_ids, customer]
        nearest_depot = self.depot_ids[int(np.argmin(dist_to_depots))]
        return nearest_depot

    def _distance_increase(self, route: List[int], node: int, pos: int) -> float:
        """
        Insert `node` into `route` at position `pos`
        (pos==0: front, pos==len(route): end),
        and return the travel distance increase.

        Depot treatment:
        - start and end depot are approximated by nearest depot
        """
        M = self.total_distance_matrix

        if not route:
            d0 = self._get_nearest_depot(node)
            return float(M[d0, node] + M[node, d0])

        node_depot = self._get_nearest_depot(node)

        if pos == 0:
            first = route[0]
            first_depot = self._get_nearest_depot(first)
            old_dist = float(M[first_depot, first])
            new_dist = float(M[node_depot, node] + M[node, first])

        elif pos == len(route):
            last = route[-1]
            last_depot = self._get_nearest_depot(last)
            old_dist = float(M[last, last_depot])
            new_dist = float(M[last, node] + M[node, node_depot])

        else:
            prev_node = route[pos - 1]
            next_node = route[pos]
            old_dist = float(M[prev_node, next_node])
            new_dist = float(M[prev_node, node] + M[node, next_node])

        return float(new_dist - old_dist)

    def _cost_increase(self, route: List[int], node: int, pos: int) -> float:
        """
        Optional exact insertion-cost increase if a route evaluator is available.
        """
        if self.evaluator is None:
            raise RuntimeError(
                "NeighborhoodNoGB.evaluator is None. "
                "Pass evaluator=planner._evaluate when creating NeighborhoodNoGB."
            )

        if not route:
            _, new_cost, *_ = self.evaluator([node])
            return float(new_cost)

        _, old_cost, *_ = self.evaluator(route)

        new_route = route.copy()
        new_route.insert(pos, node)
        _, new_cost, *_ = self.evaluator(new_route)

        return float(new_cost - old_cost)

    # ============================================================
    # Node-level: intra-route
    # ============================================================
    def relocate_node_intra_route(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None,
    ) -> RoutesByClusters:
        """
        ONE move: relocate one node within the same route.
        """
        sol = self.deep_copy_routes(routes_by_clusters)

        cands = [cid] if (cid is not None and cid in sol) else list(sol.keys())
        cands = [c for c in cands if len(sol[c]) >= 2]
        c = self._pick_cluster(cands)
        if c is None:
            return sol

        route = sol[c]
        i = int(self.rng.integers(0, len(route)))
        node = route.pop(i)

        j = int(self.rng.integers(0, len(route) + 1))
        route.insert(j, node)
        return sol

    def swap_node_intra_route(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None,
    ) -> RoutesByClusters:
        """
        ONE move: swap two nodes within the same route.
        """
        sol = self.deep_copy_routes(routes_by_clusters)

        cands = [cid] if (cid is not None and cid in sol) else list(sol.keys())
        cands = [c for c in cands if len(sol[c]) >= 2]
        c = self._pick_cluster(cands)
        if c is None:
            return sol

        route = sol[c]
        i, j = self.rng.choice(len(route), size=2, replace=False)
        i, j = int(i), int(j)
        route[i], route[j] = route[j], route[i]
        return sol

    def reverse_node_intra_route(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None,
    ) -> RoutesByClusters:
        """
        ONE move: reverse a contiguous route segment within the same route.
        """
        sol = self.deep_copy_routes(routes_by_clusters)

        cands = [cid] if (cid is not None and cid in sol) else list(sol.keys())
        cands = [c for c in cands if len(sol[c]) >= 2]
        c = self._pick_cluster(cands)
        if c is None:
            return sol

        route = sol[c]
        i = int(self.rng.integers(0, len(route) - 1))
        j = int(self.rng.integers(i + 1, len(route)))
        route[i : j + 1] = list(reversed(route[i : j + 1]))
        return sol

    # optional alias: if some old code refers to 2-opt-like move
    def node_2opt_intra_route(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None,
    ) -> RoutesByClusters:
        return self.reverse_node_intra_route(routes_by_clusters, cid=cid)

    # ============================================================
    # Node-level: inter-route
    # ============================================================
    def relocate_node_inter_route(
        self,
        routes_by_clusters: RoutesByClusters,
        *,
        src_cid: Optional[int] = None,
        dst_cid: Optional[int] = None,
    ) -> RoutesByClusters:
        """
        ONE move: move 1 node from route(src) -> route(dst),
        subject to capacity:
            load(dst) + demand(node) <= cap
        """
        sol = self.deep_copy_routes(routes_by_clusters)

        def has_any_node(c: int) -> bool:
            return c in sol and len(sol[c]) > 0

        # pick src
        if src_cid is not None:
            src_cands = [src_cid] if has_any_node(src_cid) else []
        else:
            src_cands = [c for c in sol.keys() if has_any_node(c)]
        if not src_cands:
            return sol
        src = int(self.rng.choice(src_cands))

        # pick a node in src
        pos = int(self.rng.integers(0, len(sol[src])))
        node = sol[src][pos]
        dem = self.problem.demand[node]

        # pick feasible dst
        def cap_ok(dst: int) -> bool:
            return dst != src and (self._route_load(sol[dst]) + dem) <= self.cap

        if dst_cid is not None:
            dst_cands = [dst_cid] if (dst_cid in sol and cap_ok(dst_cid)) else []
        else:
            dst_cands = [c for c in sol.keys() if cap_ok(c)]
        if not dst_cands:
            return sol
        dst = int(self.rng.choice(dst_cands))

        # commit
        node = sol[src].pop(pos)
        ins = int(self.rng.integers(0, len(sol[dst]) + 1))
        sol[dst].insert(ins, node)
        return sol

    def swap_node_inter_route(
        self,
        routes_by_clusters: RoutesByClusters,
        *,
        cid_a: Optional[int] = None,
        cid_b: Optional[int] = None,
    ) -> RoutesByClusters:
        """
        ONE move: swap 1 node between two different routes,
        subject to capacity after swap.
        """
        sol = self.deep_copy_routes(routes_by_clusters)

        def has_any_node(c: int) -> bool:
            return c in sol and len(sol[c]) > 0

        # pick A
        if cid_a is not None:
            A_cands = [cid_a] if has_any_node(cid_a) else []
        else:
            A_cands = [c for c in sol.keys() if has_any_node(c)]
        if not A_cands:
            return sol
        A = int(self.rng.choice(A_cands))

        # pick B
        if cid_b is not None:
            B_cands = [cid_b] if (cid_b != A and has_any_node(cid_b)) else []
        else:
            B_cands = [c for c in sol.keys() if c != A and has_any_node(c)]
        if not B_cands:
            return sol
        B = int(self.rng.choice(B_cands))

        posA = int(self.rng.integers(0, len(sol[A])))
        posB = int(self.rng.integers(0, len(sol[B])))

        nodeA = sol[A][posA]
        nodeB = sol[B][posB]
        demA = self.problem.demand[nodeA]
        demB = self.problem.demand[nodeB]

        loadA = self._route_load(sol[A])
        loadB = self._route_load(sol[B])

        if (loadA - demA + demB > self.cap) or (loadB - demB + demA > self.cap):
            return sol

        sol[A][posA], sol[B][posB] = sol[B][posB], sol[A][posA]
        return sol

    # ============================================================
    # Boundary destroy + repair
    # ============================================================
    def boundary_destroy(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None,
    ) -> Tuple[RoutesByClusters, List[int]]:
        """
        Destroy only:
        remove a subset of customers according to boundary proximity
        on flat routes.
        """
        sol = self.deep_copy_routes(routes_by_clusters)

        cluster_ids = [cid] if cid is not None else list(sol.keys())
        removed_all: List[int] = []

        for c in cluster_ids:
            route = sol.get(c, [])
            if not route:
                continue

            L = len(route)
            if L == 0:
                continue

            num_remove = max(1, int(L * self.removal_ratio))

            # end-biased roulette weights
            scores = np.array(
                [self._proximity(i, L) for i in range(L)],
                dtype=float,
            )

            if scores.sum() > 0:
                probs = scores / scores.sum()
            else:
                probs = np.ones(L, dtype=float) / L

            sample_size = min(num_remove, L)
            picked_idx = self.rng.choice(
                L, size=sample_size, replace=False, p=probs
            )

            # pop from back to front to avoid index shift
            removed_c: List[int] = []
            for idx in sorted([int(x) for x in picked_idx], reverse=True):
                removed_c.append(route.pop(idx))

            removed_all.extend(removed_c)

        return sol, removed_all

    def greedy_repair(
        self,
        routes_by_clusters: RoutesByClusters,
        removed_customers: List[int],
        cid: Optional[int] = None,
        sort_by_demand_desc: bool = True,
    ) -> RoutesByClusters:
        """
        Repair:
        insert removed customers back by cheapest insertion,
        with capacity feasibility on each flat route.
        """
        sol = self.deep_copy_routes(routes_by_clusters)
        if not removed_customers:
            return sol

        if sort_by_demand_desc:
            nodes = sorted(
                removed_customers,
                key=lambda x: self.problem.demand[x],
                reverse=True,
            )
        else:
            nodes = removed_customers[:]

        cluster_ids = [cid] if cid is not None else list(sol.keys())
        if not cluster_ids:
            return sol

        for node in nodes:
            dem = self.problem.demand[node]

            best_delta = float("inf")
            best_c = None
            best_pos = None

            for c in cluster_ids:
                if self._route_load(sol[c]) + dem > self.cap:
                    continue

                route = sol[c]
                for pos in range(len(route) + 1):
                    delta = float(self._distance_increase(route, node, pos))
                    if delta < best_delta:
                        best_delta = delta
                        best_c = c
                        best_pos = pos

            # no feasible insertion: skip
            if best_c is None:
                continue

            sol[best_c].insert(int(best_pos), node)

        return sol

    def relocate_boundary(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None,
    ) -> RoutesByClusters:
        destroyed, removed = self.boundary_destroy(routes_by_clusters, cid=cid)
        repaired = self.greedy_repair(destroyed, removed, cid=cid)
        return repaired

    # ============================================================
    # Unified dispatcher
    # ============================================================
    def apply(self, op, routes_by_clusters, **kwargs):
        fn = getattr(self, op)

        sig = inspect.signature(fn)
        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if k in sig.parameters
        }

        return fn(routes_by_clusters, **filtered_kwargs)

    def list_ops(self) -> List[str]:
        ops = []
        for name in dir(self):
            if name.startswith("_"):
                continue
            if name in ("apply", "list_ops", "deep_copy_routes"):
                continue
            attr = getattr(self, name)
            if callable(attr):
                ops.append(name)
        return sorted(ops)