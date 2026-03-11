import copy
import inspect
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Callable

RoutesByClusters = Dict[int, List[List[int]]]  # cid -> list of GB paths


class Neighborhood:
    """
    Neighborhood operators for GB-structured solutions.

    - Input/Output: RoutesByClusters -> RoutesByClusters
    - Each operator performs ONE move (not enumeration).
    - 'cid' semantics:
        cid=None : pick a feasible cluster randomly (if needed)
        cid=int  : apply move only within that cluster
    """

    def __init__(
        self,
        problem,

        alpha_boundary = 0.5,
        removal_ratio: float = 0.3,
        rng: Optional[np.random.Generator] = None,
        min_gb_len: int = 2,
        evaluator: Optional[Callable[[List[int]], Any]] = None,
    ):
        self.problem = problem
        self.total_distance_matrix = self.problem.distance_matrix
        self.cap = self.problem.vehicle_capacity
        self.depot_ids = self.problem.depots

        self.removal_ratio = float(removal_ratio)
        self.alpha_boundary = float(alpha_boundary)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.min_gb_len = int(min_gb_len)

        self.evaluator = evaluator

    # -----------------------
    # Helpers
    # -----------------------
    @staticmethod
    def deep_copy_routes(routes_by_clusters: RoutesByClusters) -> RoutesByClusters:
        return copy.deepcopy(routes_by_clusters)

    @staticmethod
    def _valid_clusters_for_gb_ops(sol: RoutesByClusters, cid: Optional[int], min_gb_count: int) -> List[int]:
        if cid is not None:
            return [cid] if cid in sol and len(sol[cid]) >= min_gb_count else []
        return [c for c, gbs in sol.items() if len(gbs) >= min_gb_count]

    @staticmethod
    def _proximity(idx: int, length: int) -> float:
        """归一化：越靠近两端越大（论文的 prox 定义）"""
        if length <= 1:
            return 0.0
        return 1.0 - min(idx, length - 1 - idx) / (length - 1)
    
    def _pick_cluster(self, cands: List[int]) -> Optional[int]:
        if not cands:
            return None
        return int(self.rng.choice(cands))

    def _cluster_load(self, gbs: List[List[int]]) -> float:
        # total demand of all nodes in all GBs
        tot = 0.0
        for gb in gbs:
            for n in gb:
                tot += self.problem.demand[n]
        return tot

    def _gb_load(self, gb: List[int]) -> float:
        return sum(self.problem.demand[n] for n in gb)

    def _any_node_positions(self, gbs: List[List[int]]) -> List[Tuple[int, int]]:
        # list of (gb_idx, pos) for all nodes
        pos = []
        for gi, gb in enumerate(gbs):
            for pi in range(len(gb)):
                pos.append((gi, pi))
        return pos


    # -----------------------
    # Boundary destroy Helpers
    # -----------------------
    @staticmethod
    def _flatten_gbs(gbs: List[List[int]]) -> List[int]:
        return [n for gb in gbs for n in gb]

    @staticmethod
    def _flatpos_to_gbpos(gbs: List[List[int]], flat_pos: int) -> Tuple[int, int]:
        """
        将 flat_pos 映射到 (gb_idx, pos_in_gb)
        flat_pos 含义：插入到 flat_route[flat_pos] 之前；flat_pos==len(flat) 表示插到末尾
        """
        if not gbs:
            return 0, 0

        cur = 0
        for gi, gb in enumerate(gbs):
            L = len(gb)
            # 插入位置落在该 GB 的可插区间 [cur, cur+L]
            if flat_pos <= cur + L:
                return gi, flat_pos - cur
            cur += L

        return len(gbs) - 1, len(gbs[-1])

    def _get_nearest_depot(self, customer):
        """
        返回距离 customer 最近的 depot id。
        依赖：self.total_distance_matrix 支持用 depot_ids 行索引到 customer 列。
        """
        dist_to_depots = self.total_distance_matrix[self.depot_ids, customer]
        nearest_depot = self.depot_ids[int(np.argmin(dist_to_depots))]
        return nearest_depot

    def _distance_increase(self, route: List[int], node: int, pos: int) -> float:
        """
        将 node 插入 route 的 pos 位置（pos==0 表示最前；pos==len(route) 表示末尾），
        返回距离增量（包含首尾 depot，且 depot 取“最近 depot”近似）。

        注意：这里的 route 应该是“整条车路线”的序列（在本类中就是 cluster 的扁平路线）。
        """
        M = self.total_distance_matrix

        if not route:
            # 空路径：old = 0，new = depot(node)→node→depot(node)
            d0 = self._get_nearest_depot(node)
            new_dist = float(M[d0, node] + M[node, d0])
            return new_dist  # old=0

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
        if self.evaluator is None:
            raise RuntimeError("Neighborhood.evaluator is None. Pass evaluator=planner._evaluate when creating Neighborhood.")

        if not route:
            _, new_cost, *_ = self.evaluator([node])
            return float(new_cost)

        _, old_cost, *_ = self.evaluator(route)

        new_route = route.copy()
        new_route.insert(pos, node)
        _, new_cost, *_ = self.evaluator(new_route)

        return float(new_cost - old_cost)




    # ============================================================
    # Node-level
    # ============================================================

    # 1) NodeRelocate: same route, intra-GB relocate
    def relocate_node_intra_gb(self, routes_by_clusters: RoutesByClusters, cid: Optional[int] = None) -> RoutesByClusters:
        """ONE random relocate (1-opt) inside a randomly chosen GB of a cluster."""
        sol = self.deep_copy_routes(routes_by_clusters)
        cands = self._valid_clusters_for_gb_ops(sol, cid, min_gb_count=1)
        c = self._pick_cluster(cands)
        if c is None:
            return sol

        gbs = sol[c]
        gb_indices = [i for i, gb in enumerate(gbs)]
        if not gb_indices:
            return sol

        gi = int(self.rng.choice(gb_indices))
        gb = gbs[gi]
        L = len(gb)

        i = int(self.rng.integers(0, L))
        node = gb.pop(i)

        newL = len(gb)
        j = int(self.rng.integers(0, newL + 1))
        gb.insert(j, node)
        return sol

    # 2) NodeRelocate: same route, across ANY two GBs 
    def relocate_node_inter_gb(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None,
        *,
        protect_min_len: bool = False,
    ) -> RoutesByClusters:
        """
        ONE move: pick a node from one GB and insert into another GB within the SAME cluster (ANY GB pair).
        """
        sol = self.deep_copy_routes(routes_by_clusters)
        cands = self._valid_clusters_for_gb_ops(sol, cid, min_gb_count=2)
        c = self._pick_cluster(cands)
        if c is None:
            return sol

        gbs = sol[c]

        # src GB candidates
        src_candidates = []
        for gi, gb in enumerate(gbs):
            if protect_min_len:
                if len(gb) > self.min_gb_len:
                    src_candidates.append(gi)
            else:
                if len(gb) > 0:
                    src_candidates.append(gi)
        if not src_candidates:
            return sol

        src_gi = int(self.rng.choice(src_candidates))
        dst_gi = int(self.rng.choice([i for i in range(len(gbs)) if i != src_gi]))

        src = gbs[src_gi]
        dst = gbs[dst_gi]

        i = int(self.rng.integers(0, len(src)))
        node = src.pop(i)

        j = int(self.rng.integers(0, len(dst) + 1))
        dst.insert(j, node)

        if len(src) == 0:
            gbs.pop(src_gi)
        return sol

    # 3) NodeRelocate: inter-route relocate (Shift(1,0)) with capacity check 
    def relocate_node_inter_route(
        self,
        routes_by_clusters: RoutesByClusters,
        *,
        src_cid: Optional[int] = None,
        dst_cid: Optional[int] = None,
        protect_min_len: bool = False,
    ) -> RoutesByClusters:
        """
        ONE move: move 1 node from Route(src cluster) -> Route(dst cluster),
        subject to: load(dst) + demand(node) <= cap.
        """
        sol = self.deep_copy_routes(routes_by_clusters)

        def has_any_node(c: int) -> bool:
            return c in sol and any(len(gb) > 0 for gb in sol[c])

        # pick src
        if src_cid is not None:
            src_cands = [src_cid] if has_any_node(src_cid) else []
        else:
            src_cands = [c for c in sol.keys() if has_any_node(c)]
        if not src_cands:
            return sol
        src = int(self.rng.choice(src_cands))

        # pick a removable node in src
        src_gbs = sol[src]
        removable_gb = []
        for gi, gb in enumerate(src_gbs):
            if protect_min_len:
                if len(gb) > self.min_gb_len:
                    removable_gb.append(gi)
            else:
                if len(gb) > 0:
                    removable_gb.append(gi)
        if not removable_gb:
            return sol

        src_gi = int(self.rng.choice(removable_gb))
        src_gb = src_gbs[src_gi]
        pos = int(self.rng.integers(0, len(src_gb)))
        node = src_gb[pos]
        dem = self.problem.demand[node]

        # pick dst candidates with capacity feasibility
        def cap_ok(dst: int) -> bool:
            if dst == src:
                return False
            return (self._cluster_load(sol[dst]) + dem) <= self.cap

        if dst_cid is not None:
            dst_cands = [dst_cid] if (dst_cid in sol and cap_ok(dst_cid)) else []
        else:
            dst_cands = [c for c in sol.keys() if c != src and cap_ok(c)]
        if not dst_cands:
            return sol
        dst = int(self.rng.choice(dst_cands))

        # commit move
        node = src_gb.pop(pos)
        if len(src_gb) == 0:
            src_gbs.pop(src_gi)

        dst_gbs = sol[dst]
        if len(dst_gbs) == 0:
            dst_gbs.append([])

        dst_gi = int(self.rng.integers(0, len(dst_gbs)))
        dst_gb = dst_gbs[dst_gi]
        ins = int(self.rng.integers(0, len(dst_gb) + 1))
        dst_gb.insert(ins, node)
        return sol




    # # ------------------------------------------------------------
    # # NodeExchange
    # # ------------------------------------------------------------
    # # 1) same route, intra-GB exchange
    # # 2) inter-route exchange (Swap(1,1)) with capacity check
    def swap_node_intra_gb(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None,
    ) -> RoutesByClusters:
        """ONE move: swap two nodes inside the same GB (intra-GB)."""
        sol = self.deep_copy_routes(routes_by_clusters)

        # pick a cluster that has at least one GB with >=2 nodes
        cands = self._valid_clusters_for_gb_ops(sol, cid, min_gb_count=1)
        c = self._pick_cluster(cands)
        if c is None:
            return sol

        gbs = sol[c]
        gb_idx = [gi for gi, gb in enumerate(gbs) if len(gb) >= 2]
        if not gb_idx:
            return sol

        gi = int(self.rng.choice(gb_idx))
        gb = gbs[gi]

        i, j = self.rng.choice(len(gb), size=2, replace=False)
        i, j = int(i), int(j)
        gb[i], gb[j] = gb[j], gb[i]
        return sol

    def swap_node_inter_gb(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None,
    ) -> RoutesByClusters:
        """ONE move: swap two nodes anywhere in the same route/cluster (intra-route)."""
        sol = self.deep_copy_routes(routes_by_clusters)

        cands = self._valid_clusters_for_gb_ops(sol, cid, min_gb_count=1)
        c = self._pick_cluster(cands)
        if c is None:
            return sol

        gbs = sol[c]
        positions = self._any_node_positions(gbs)
        if len(positions) < 2:
            return sol

        (g1, p1), (g2, p2) = self.rng.choice(positions, size=2, replace=False)
        g1, p1, g2, p2 = int(g1), int(p1), int(g2), int(p2)

        gbs[g1][p1], gbs[g2][p2] = gbs[g2][p2], gbs[g1][p1]
        return sol

    def swap_node_inter_route(
        self,
        routes_by_clusters: RoutesByClusters,
        *,
        cid_a: Optional[int] = None,
        cid_b: Optional[int] = None,
    ) -> RoutesByClusters:
        """ONE move: swap 1 node between two different routes, subject to capacity."""
        sol = self.deep_copy_routes(routes_by_clusters)

        def has_any_node(c: int) -> bool:
            return c in sol and any(len(gb) > 0 for gb in sol[c])

        # pick route A
        if cid_a is not None:
            A_cands = [cid_a] if has_any_node(cid_a) else []
        else:
            A_cands = [c for c in sol.keys() if has_any_node(c)]
        if not A_cands:
            return sol
        A = int(self.rng.choice(A_cands))

        # pick route B
        if cid_b is not None:
            B_cands = [cid_b] if (cid_b != A and has_any_node(cid_b)) else []
        else:
            B_cands = [c for c in sol.keys() if c != A and has_any_node(c)]
        if not B_cands:
            return sol
        B = int(self.rng.choice(B_cands))

        gbsA, gbsB = sol[A], sol[B]

        # pick random node from A
        candA = [gi for gi, gb in enumerate(gbsA) if len(gb) > 0]
        candB = [gi for gi, gb in enumerate(gbsB) if len(gb) > 0]
        if not candA or not candB:
            return sol

        giA = int(self.rng.choice(candA))
        giB = int(self.rng.choice(candB))
        gbA, gbB = gbsA[giA], gbsB[giB]

        posA = int(self.rng.integers(0, len(gbA)))
        posB = int(self.rng.integers(0, len(gbB)))

        nodeA, nodeB = gbA[posA], gbB[posB]
        demA = self.problem.demand[nodeA]
        demB = self.problem.demand[nodeB]

        loadA = self._cluster_load(gbsA)
        loadB = self._cluster_load(gbsB)

        if (loadA - demA + demB > self.cap) or (loadB - demB + demA > self.cap):
            return sol

        gbA[posA], gbB[posB] = gbB[posB], gbA[posA]
        return sol


    # ------------------------------------------------------------
    # Reverse / 2-opt
    # ------------------------------------------------------------
    def node_2opt_intra_gb(self, routes_by_clusters: RoutesByClusters, cid: Optional[int] = None) -> RoutesByClusters:
        """ONE random 2-opt (segment reverse) inside a randomly chosen GB of a cluster."""
        sol = self.deep_copy_routes(routes_by_clusters)
        cands = self._valid_clusters_for_gb_ops(sol, cid, min_gb_count=1)
        c = self._pick_cluster(cands)
        if c is None:
            return sol

        gbs = sol[c]
        gb_indices = [i for i, gb in enumerate(gbs) if len(gb) >= 4]
        if not gb_indices:
            return sol

        gi = int(self.rng.choice(gb_indices))
        gb = gbs[gi]
        L = len(gb)

        i = int(self.rng.integers(0, L - 1))
        j = int(self.rng.integers(i + 1, L))
        gb[i : j + 1] = list(reversed(gb[i : j + 1]))
        return sol
    
    def reverse_node_intra_gb(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None,
    ) -> RoutesByClusters:
        """
        ONE move: reverse the ENTIRE node order of a GB (full reverse).

        Example:
            [n1, n2, n3, n4] -> [n4, n3, n2, n1]
        """
        sol = self.deep_copy_routes(routes_by_clusters)

        # need at least one GB
        cands = self._valid_clusters_for_gb_ops(sol, cid, min_gb_count=1)
        c = self._pick_cluster(cands)
        if c is None:
            return sol

        gbs = sol[c]

        # pick a GB with length >= 2 (otherwise reverse is meaningless)
        gb_indices = [i for i, gb in enumerate(gbs) if len(gb) >= 2]
        if not gb_indices:
            return sol

        gi = int(self.rng.choice(gb_indices))
        gbs[gi].reverse()

        return sol





    # ============================================================
    # GB-level
    # ============================================================

    # 1) GBRelocate: same route (已有)
    def relocate_gb_intra_route(self, routes_by_clusters: RoutesByClusters, cid: Optional[int] = None) -> RoutesByClusters:
        """ONE random relocate of a GB within a cluster."""
        sol = self.deep_copy_routes(routes_by_clusters)
        cands = self._valid_clusters_for_gb_ops(sol, cid, min_gb_count=2)
        c = self._pick_cluster(cands)
        if c is None:
            return sol

        gbs = sol[c]
        i = int(self.rng.integers(0, len(gbs)))
        gb = gbs.pop(i)
        j = int(self.rng.integers(0, len(gbs) + 1))
        gbs.insert(j, gb)
        return sol

    # 2) GBRelocate: inter-route (need cap check)
    def relocate_gb_inter_route(
        self,
        routes_by_clusters: RoutesByClusters,
        *,
        src_cid: Optional[int] = None,
        dst_cid: Optional[int] = None,
    ) -> RoutesByClusters:
        """
        ONE move: move 1 GB from Route(src) -> Route(dst),
        subject to: load(dst) + load(GB) <= cap.
        """
        sol = self.deep_copy_routes(routes_by_clusters)

        # pick src with at least 1 GB
        src_cands = [src_cid] if (src_cid is not None and src_cid in sol and len(sol[src_cid]) > 0) else []
        if src_cid is None:
            src_cands = [c for c, gbs in sol.items() if len(gbs) > 0]
        if not src_cands:
            return sol
        src = int(self.rng.choice(src_cands))

        src_gbs = sol[src]
        gi = int(self.rng.integers(0, len(src_gbs)))
        gb = src_gbs[gi]
        gb_dem = self._gb_load(gb)

        # pick dst feasible by capacity
        def cap_ok(dst: int) -> bool:
            if dst == src:
                return False
            return (self._cluster_load(sol[dst]) + gb_dem) <= self.cap

        if dst_cid is not None:
            dst_cands = [dst_cid] if (dst_cid in sol and cap_ok(dst_cid)) else []
        else:
            dst_cands = [c for c in sol.keys() if c != src and cap_ok(c)]
        if not dst_cands:
            return sol
        dst = int(self.rng.choice(dst_cands))

        # commit move
        gb = src_gbs.pop(gi)
        sol[dst].insert(int(self.rng.integers(0, len(sol[dst]) + 1)), gb)
        return sol

    # 1) GBExchange: same route
    def swap_gb_intra_route(self, routes_by_clusters: RoutesByClusters, cid: Optional[int] = None) -> RoutesByClusters:
        """ONE random swap of two GBs within a cluster."""
        sol = self.deep_copy_routes(routes_by_clusters)
        cands = self._valid_clusters_for_gb_ops(sol, cid, min_gb_count=2)
        c = self._pick_cluster(cands)
        if c is None:
            return sol

        gbs = sol[c]
        i, j = self.rng.choice(len(gbs), size=2, replace=False)
        i, j = int(i), int(j)
        gbs[i], gbs[j] = gbs[j], gbs[i]
        return sol

    # 2) GBExchange: inter-route (need cap check)
    def swap_gb_inter_route(
        self,
        routes_by_clusters: RoutesByClusters,
        *,
        cid_a: Optional[int] = None,
        cid_b: Optional[int] = None,
    ) -> RoutesByClusters:
        """
        ONE move: swap 1 GB between Route A and Route B,
        subject to capacity on both sides after swap.
        """
        sol = self.deep_copy_routes(routes_by_clusters)

        # pick A
        if cid_a is not None:
            A_cands = [cid_a] if (cid_a in sol and len(sol[cid_a]) > 0) else []
        else:
            A_cands = [c for c, gbs in sol.items() if len(gbs) > 0]
        if not A_cands:
            return sol
        A = int(self.rng.choice(A_cands))

        # pick B
        if cid_b is not None:
            B_cands = [cid_b] if (cid_b in sol and cid_b != A and len(sol[cid_b]) > 0) else []
        else:
            B_cands = [c for c, gbs in sol.items() if c != A and len(gbs) > 0]
        if not B_cands:
            return sol
        B = int(self.rng.choice(B_cands))

        gbsA = sol[A]
        gbsB = sol[B]

        giA = int(self.rng.integers(0, len(gbsA)))
        giB = int(self.rng.integers(0, len(gbsB)))

        gbA = gbsA[giA]
        gbB = gbsB[giB]
        demA = self._gb_load(gbA)
        demB = self._gb_load(gbB)

        loadA = self._cluster_load(gbsA)
        loadB = self._cluster_load(gbsB)

        # after swap:
        if (loadA - demA + demB > self.cap) or (loadB - demB + demA > self.cap):
            return sol

        # commit swap
        gbsA[giA], gbsB[giB] = gbsB[giB], gbsA[giA]
        return sol


    # 2) relocate: inter-route (need cap check)
    def boundary_destroy(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None,
    ) -> Tuple[RoutesByClusters, List[int]]:
        """
        只做 destroy：按边界概率删除一批客户（论文版：GB边界+cluster边界融合 + roulette-wheel）。
        返回：new_routes_by_clusters, removed_customers
        """
        sol = self.deep_copy_routes(routes_by_clusters)

        cluster_ids = [cid] if cid is not None else list(sol.keys())
        removed_all: List[int] = []

        for c in cluster_ids:
            gbs = sol.get(c, [])
            if not gbs:
                continue

            full = [n for gb in gbs for n in gb]
            if len(full) == 0:
                continue

            num_remove = max(1, int(len(full) * self.removal_ratio))

            # cluster 内位置映射
            pos_in_cluster = {nid: idx for idx, nid in enumerate(full)}
            L_cl = len(full)

            candidates = []  # (score, nid)
            for gb in gbs:
                L_gb = len(gb)
                for i, nid in enumerate(gb):
                    prox_gb = self._proximity(i, L_gb)
                    j = pos_in_cluster.get(nid, 0)
                    prox_cl = self._proximity(j, L_cl)

                    score = self.alpha_boundary * prox_gb + (1.0 - self.alpha_boundary) * prox_cl
                    candidates.append((score, nid))

            if not candidates:
                continue

            scores = np.array([s for s, _ in candidates], dtype=float)
            if scores.sum() > 0:
                probs = scores / scores.sum()
            else:
                probs = np.ones(len(candidates), dtype=float) / len(candidates)

            sample_size = min(num_remove, len(candidates))
            picked_idx = self.rng.choice(len(candidates), size=sample_size, replace=False, p=probs)
            to_remove = {candidates[i][1] for i in picked_idx}

            # 删除（带 GB 最短长度保护）
            new_gbs: List[List[int]] = []
            removed_c: List[int] = []

            for gb in gbs:
                new_gb = [n for n in gb if n not in to_remove]
                if len(new_gb) < self.min_gb_len:
                    # 保底：不删这个 GB（否则 GB 太短）
                    new_gbs.append(gb[:])
                else:
                    removed_local = [n for n in gb if n in to_remove]
                    removed_c.extend(removed_local)
                    new_gbs.append(new_gb)

            sol[c] = new_gbs
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
        Repair：把 removed_customers 用 cheapest insertion 插回去（含 depot 首尾增量）。
        - cluster = 一辆车：容量约束按 cluster 总载重
        - 插入位置在 cluster 扁平路线 flat 上评估 delta（_distance_increase 自动处理 pos=0/末尾的 depot）
        - 插入后映射回某个 GB 内部插入，保持 GB 结构
        cid:
          - 指定 cid：只插回该 cluster
          - cid=None：跨 cluster 找全局最便宜且容量可行的位置
        """
        sol = self.deep_copy_routes(routes_by_clusters)
        if not removed_customers:
            return sol

        # 插入顺序：大需求优先更稳
        if sort_by_demand_desc:
            nodes = sorted(removed_customers, key=lambda x: self.problem.demand[x], reverse=True)
        else:
            nodes = removed_customers[:]

        # 预计算每个 cluster 当前载重
        cl_load = {c: self._cluster_load(gbs) for c, gbs in sol.items()}

        cluster_ids = [cid] if cid is not None else list(sol.keys())
        if not cluster_ids:
            return sol
        
        def _new_cluster_id(existing_ids) -> int:
            if not existing_ids:
                return 0
            return int(max(existing_ids)) + 1

        for node in nodes:
            dem = self.problem.demand[node]

            best_delta = float("inf")
            best_c = None
            best_pos = None

            for c in cluster_ids:
                gbs = sol.get(c, [])
                if gbs is None:
                    continue

                # 容量按 cluster 判断
                if cl_load.get(c, 0.0) + dem > self.cap:
                    continue

                flat = self._flatten_gbs(gbs)
                L = len(flat)

                # 全位置枚举（含 pos=0/pos=L，自动含 depot 增量）
                for pos in range(L + 1):
                    delta = float(self._distance_increase(flat, node, pos))
                    if delta < best_delta:
                        best_delta = delta
                        best_c = c
                        best_pos = pos

            if best_c is None:
                # # ✅ 兜底：新建 cluster（新车），并单独成一个 GB
                # new_c = _new_cluster_id(sol.keys() if cid is None else [cid])
                # sol[new_c] = [[int(node)]]
                # cl_load[new_c] = cl_load.get(new_c, 0.0) + dem

                # # 若 cid=None，需要把新簇加入后续可选集合（让后续点也可插入该新簇）
                # if cid is None:
                #     cluster_ids.append(new_c)
                continue

            # 落地插入：把 flat_pos 映射回具体某个 GB 内
            if best_c not in sol:
                sol[best_c] = []

            # 如果 cluster 没有 GB 容器，按需创建（一般不应发生）
            if len(sol[best_c]) == 0:
                sol[best_c].append([node])
                cl_load[best_c] = cl_load.get(best_c, 0.0) + dem
                continue

            gi, pos_in_gb = self._flatpos_to_gbpos(sol[best_c], int(best_pos))
            sol[best_c][gi].insert(int(pos_in_gb), node)
            cl_load[best_c] = cl_load.get(best_c, 0.0) + dem

        return sol

    def relocate_boundary(self, routes_by_clusters):
        destroyed, removed = self.boundary_destroy(routes_by_clusters, cid=None)
        repaired = self.greedy_repair(destroyed, removed, cid=None)
        return repaired
    

    def merge_gb_intra_route(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None,
    ) -> RoutesByClusters:
        """
        Merge two adjacent GBs within the same route (cluster).
        """
        sol = self.deep_copy_routes(routes_by_clusters)

        # 需要至少 2 个 GB
        cands = [cid] if cid is not None else [c for c in sol if len(sol[c]) >= 2]
        if not cands:
            return sol

        c = int(self.rng.choice(cands))
        gbs = sol[c]
        if len(gbs) < 2:
            return sol

        # 随机选一对相邻 GB
        i = int(self.rng.integers(0, len(gbs) - 1))

        # 直接聚合
        gbs[i] = gbs[i] + gbs[i + 1]
        del gbs[i + 1]

        return sol

    def gb_2opt_intra_route(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None,
        min_segment_gb: int = 2,
    ) -> RoutesByClusters:
        """
        ONE random 2-opt on GB sequence within a cluster route.
        Treat each GB as a super-node and reverse a contiguous GB segment.

        Example:
        [GB0, GB1, GB2, GB3, GB4] with i=1, j=3
        -> reverse GB1..GB3
        [GB0, GB3, GB2, GB1, GB4]

        Args:
        cid: specify cluster id; None means pick any feasible cluster
        min_segment_gb: minimum number of GBs in the reversed segment (>=2 typical)
        """
        sol = self.deep_copy_routes(routes_by_clusters)

        # Need at least 4 GBs to make a meaningful 2-opt that changes structure
        cands = [cid] if cid is not None else [c for c in sol.keys() if len(sol[c]) >= 4]
        if not cands:
            return sol

        c = int(self.rng.choice(cands))
        gbs = sol[c]
        n = len(gbs)
        if n < 4:
            return sol

        # pick two cut indices i < j, reverse gbs[i:j+1]
        # enforce segment length
        # i in [0, n-2], j in [i+1, n-1]
        # and (j - i + 1) >= min_segment_gb
        if min_segment_gb < 2:
            min_segment_gb = 2

        # Try a few random attempts to satisfy min_segment_gb
        for _ in range(20):
            i = int(self.rng.integers(0, n - 1))
            j = int(self.rng.integers(0, n - 1))
            if i == j:
                continue
            if i > j:
                i, j = j, i
            if (j - i + 1) < min_segment_gb:
                continue
            gbs[i:j + 1] = list(reversed(gbs[i:j + 1]))
            return sol

        return sol





        # ============================================================
    # Rewritten from code1 (keep old names for compatibility)
    # ============================================================

    # -----------------------
    # Node-level
    # -----------------------

    def node_intra_gb_relocate(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None
    ) -> RoutesByClusters:
        """ONE random relocate (1-opt) inside a randomly chosen GB of a cluster."""
        sol = self.deep_copy_routes(routes_by_clusters)
        cands = self._valid_clusters_for_gb_ops(sol, cid, min_gb_count=1)
        c = self._pick_cluster(cands)
        if c is None:
            return sol

        gbs = sol[c]
        gb_indices = [i for i, gb in enumerate(gbs) if len(gb) >= 3]
        if not gb_indices:
            return sol

        gi = int(self.rng.choice(gb_indices))
        gb = gbs[gi]
        L = len(gb)

        i = int(self.rng.integers(0, L))
        node = gb.pop(i)

        newL = len(gb)
        j = int(self.rng.integers(0, newL + 1))
        gb.insert(j, node)
        return sol

    def node_intra_route_relocate_across_any_gb(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None,
        *,
        protect_min_len: bool = True,
        drop_empty_gb: bool = True,
    ) -> RoutesByClusters:
        """
        ONE move: pick a node from one GB and insert into another GB within the SAME cluster (ANY GB pair).
        """
        sol = self.deep_copy_routes(routes_by_clusters)
        cands = self._valid_clusters_for_gb_ops(sol, cid, min_gb_count=2)
        c = self._pick_cluster(cands)
        if c is None:
            return sol

        gbs = sol[c]

        src_candidates = []
        for gi, gb in enumerate(gbs):
            if protect_min_len:
                if len(gb) > self.min_gb_len:
                    src_candidates.append(gi)
            else:
                if len(gb) > 0:
                    src_candidates.append(gi)
        if not src_candidates:
            return sol

        src_gi = int(self.rng.choice(src_candidates))
        dst_gi = int(self.rng.choice([i for i in range(len(gbs)) if i != src_gi]))

        src = gbs[src_gi]
        dst = gbs[dst_gi]

        i = int(self.rng.integers(0, len(src)))
        node = src.pop(i)

        j = int(self.rng.integers(0, len(dst) + 1))
        dst.insert(j, node)

        if drop_empty_gb and len(src) == 0:
            gbs.pop(src_gi)
        return sol

    def node_intra_gb_2opt(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None
    ) -> RoutesByClusters:
        """ONE random 2-opt (segment reverse) inside a randomly chosen GB of a cluster."""
        sol = self.deep_copy_routes(routes_by_clusters)
        cands = self._valid_clusters_for_gb_ops(sol, cid, min_gb_count=1)
        c = self._pick_cluster(cands)
        if c is None:
            return sol

        gbs = sol[c]
        gb_indices = [i for i, gb in enumerate(gbs) if len(gb) >= 4]
        if not gb_indices:
            return sol

        gi = int(self.rng.choice(gb_indices))
        gb = gbs[gi]
        L = len(gb)

        i = int(self.rng.integers(0, L - 1))
        j = int(self.rng.integers(i + 1, L))
        gb[i:j + 1] = list(reversed(gb[i:j + 1]))
        return sol

    def node_intra_gb_reverse(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None,
    ) -> RoutesByClusters:
        """
        ONE move: reverse the ENTIRE node order of a GB (full reverse).
        """
        sol = self.deep_copy_routes(routes_by_clusters)

        cands = self._valid_clusters_for_gb_ops(sol, cid, min_gb_count=1)
        c = self._pick_cluster(cands)
        if c is None:
            return sol

        gbs = sol[c]
        gb_indices = [i for i, gb in enumerate(gbs) if len(gb) >= 2]
        if not gb_indices:
            return sol

        gi = int(self.rng.choice(gb_indices))
        gbs[gi].reverse()

        return sol

    # -----------------------
    # GB-level
    # -----------------------

    def gb_relocate(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None
    ) -> RoutesByClusters:
        """ONE random relocate of a GB within a cluster."""
        sol = self.deep_copy_routes(routes_by_clusters)
        cands = self._valid_clusters_for_gb_ops(sol, cid, min_gb_count=2)
        c = self._pick_cluster(cands)
        if c is None:
            return sol

        gbs = sol[c]
        i = int(self.rng.integers(0, len(gbs)))
        gb = gbs.pop(i)
        j = int(self.rng.integers(0, len(gbs) + 1))
        gbs.insert(j, gb)
        return sol

    def gb_swap(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None
    ) -> RoutesByClusters:
        """ONE random swap of two GBs within a cluster."""
        sol = self.deep_copy_routes(routes_by_clusters)
        cands = self._valid_clusters_for_gb_ops(sol, cid, min_gb_count=2)
        c = self._pick_cluster(cands)
        if c is None:
            return sol

        gbs = sol[c]
        i, j = self.rng.choice(len(gbs), size=2, replace=False)
        i, j = int(i), int(j)
        gbs[i], gbs[j] = gbs[j], gbs[i]
        return sol

    def gb_reverse(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None
    ) -> RoutesByClusters:
        """ONE random reverse of a contiguous GB segment within a cluster."""
        sol = self.deep_copy_routes(routes_by_clusters)
        cands = self._valid_clusters_for_gb_ops(sol, cid, min_gb_count=3)
        c = self._pick_cluster(cands)
        if c is None:
            return sol

        gbs = sol[c]
        i = int(self.rng.integers(0, len(gbs) - 1))
        j = int(self.rng.integers(i + 1, len(gbs)))
        gbs[i:j + 1] = list(reversed(gbs[i:j + 1]))
        return sol

    # -----------------------
    # Boundary relocate (destroy + repair)
    # -----------------------

    def boundary_relocate(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None
    ) -> RoutesByClusters:
        destroyed, removed = self.boundary_destroy(routes_by_clusters, cid=cid)
        repaired = self.greedy_repair(destroyed, removed, cid=cid)
        return repaired

    

    # def relocate_node_intra_route(
    #     self,
    #     routes_by_clusters: RoutesByClusters,
    #     cid: Optional[int] = None,
    # ) -> RoutesByClusters:
    #     """ONE move: relocate one node within the same route (intra-route)."""
    #     sol = self.deep_copy_routes(routes_by_clusters)

    #     cands = list(sol.keys()) if cid is None else ([cid] if cid in sol else [])
    #     if not cands:
    #         return sol
    #     c = int(self.rng.choice(cands))

    #     gbs = sol[c]
    #     if not gbs or len(gbs[0]) < 2:
    #         return sol

    #     route = gbs[0]
    #     L = len(route)

    #     i = int(self.rng.integers(0, L))
    #     node = route.pop(i)

    #     # choose an insertion position different from i (avoid no-op as much as possible)
    #     j = int(self.rng.integers(0, L - 1))
    #     if j >= i:
    #         j += 1
    #     route.insert(j, node)

    #     return sol

    # def swap_node_intra_route(
    #     self,
    #     routes_by_clusters: RoutesByClusters,
    #     cid: Optional[int] = None,
    # ) -> RoutesByClusters:
    #     """ONE move: swap two nodes within the same route (intra-route)."""
    #     sol = self.deep_copy_routes(routes_by_clusters)

    #     # pick a cluster (route)
    #     cands = list(sol.keys()) if cid is None else ([cid] if cid in sol else [])
    #     if not cands:
    #         return sol
    #     c = int(self.rng.choice(cands))

    #     gbs = sol[c]
    #     if not gbs or len(gbs[0]) < 2:
    #         return sol

    #     # withoutGB: single GB holds the whole route
    #     route = gbs[0]
    #     i, j = self.rng.choice(len(route), size=2, replace=False)
    #     i, j = int(i), int(j)
    #     route[i], route[j] = route[j], route[i]
    #     return sol

    # def reverse_node_intra_route(
    #     self,
    #     routes_by_clusters: RoutesByClusters,
    #     cid: Optional[int] = None,
    # ) -> RoutesByClusters:
    #     """Alias for withoutGB: reverse node order within the same route (intra-route)."""
    #     return self.reverse_node_intra_gb(routes_by_clusters, cid=cid)





    # # =========================
    # # Flat-route helpers (NEW)
    # # =========================
    # def _cluster_to_flat_and_cuts(self, gbs: List[List[int]]) -> Tuple[List[int], List[int]]:
    #     """
    #     gbs: [gb1, gb2, ...]
    #     returns:
    #     flat: concatenated nodes
    #     cuts: cumulative end positions of each GB in flat (e.g., [3, 7, 10])
    #     """
    #     flat = [n for gb in gbs for n in gb]
    #     cuts = []
    #     s = 0
    #     for gb in gbs:
    #         s += len(gb)
    #         cuts.append(s)
    #     return flat, cuts


    # def _sanitize_cuts(self, cuts: List[int], L: int) -> List[int]:
    #     """Ensure cuts are strictly increasing, within [0, L], and last == L."""
    #     if L <= 0:
    #         return []

    #     # keep valid
    #     cuts = [int(c) for c in cuts if 0 <= int(c) <= L]
    #     if not cuts:
    #         return [L]

    #     # sort & unique
    #     cuts = sorted(set(cuts))

    #     # ensure last == L
    #     if cuts[-1] != L:
    #         # remove any > L already; now append L
    #         cuts = [c for c in cuts if c < L] + [L]

    #     # remove 0 (would create empty GB at start)
    #     cuts = [c for c in cuts if c > 0]
    #     if not cuts or cuts[-1] != L:
    #         cuts = [c for c in cuts if c < L] + [L]

    #     return cuts


    # def _flat_and_cuts_to_cluster(self, flat: List[int], cuts: List[int]) -> List[List[int]]:
    #     """
    #     Cut flat into GBs by cuts, then repair:
    #     - remove empty GBs
    #     - merge GBs shorter than min_gb_len (to keep GB-ops stable)
    #     """
    #     L = len(flat)
    #     if L == 0:
    #         return []

    #     cuts = self._sanitize_cuts(cuts, L)

    #     gbs: List[List[int]] = []
    #     prev = 0
    #     for c in cuts:
    #         gbs.append(flat[prev:c])
    #         prev = c

    #     # remove empties
    #     gbs = [gb for gb in gbs if len(gb) > 0]
    #     if not gbs:
    #         return [flat[:]]

    #     # merge too-short GBs
    #     i = 0
    #     while i < len(gbs) and len(gbs) > 1:
    #         if len(gbs[i]) < self.min_gb_len:
    #             if i == 0:
    #                 gbs[1] = gbs[i] + gbs[1]
    #                 gbs.pop(i)
    #             elif i == len(gbs) - 1:
    #                 gbs[i - 1] = gbs[i - 1] + gbs[i]
    #                 gbs.pop(i)
    #                 i -= 1
    #             else:
    #                 # merge into the longer neighbor
    #                 if len(gbs[i - 1]) >= len(gbs[i + 1]):
    #                     gbs[i - 1] = gbs[i - 1] + gbs[i]
    #                     gbs.pop(i)
    #                     i -= 1
    #                 else:
    #                     gbs[i + 1] = gbs[i] + gbs[i + 1]
    #                     gbs.pop(i)
    #         else:
    #             i += 1

    #     return gbs


    # def _cuts_after_remove(self, cuts: List[int], remove_pos: int) -> List[int]:
    #     """If a node at flat index remove_pos is removed, all cut positions > remove_pos shift -1."""
    #     new_cuts = []
    #     for c in cuts:
    #         c2 = c - 1 if remove_pos < c else c
    #         new_cuts.append(c2)
    #     return new_cuts


    # def _cuts_after_insert(self, cuts: List[int], insert_pos: int) -> List[int]:
    #     """If a node is inserted at flat index insert_pos, all cut positions >= insert_pos shift +1."""
    #     new_cuts = []
    #     for c in cuts:
    #         c2 = c + 1 if insert_pos <= c else c
    #         new_cuts.append(c2)
    #     return new_cuts


    # # # ============================================================
    # # # 你要的四个算子（flat-route 版）
    # # # ============================================================

    # def relocate_node_intra_route(
    #     self,
    #     routes_by_clusters: RoutesByClusters,
    #     cid: Optional[int] = None,
    # ) -> RoutesByClusters:
    #     """
    #     ONE move: relocate 1 node within the SAME route/cluster on the flattened route.
    #     After move, rebuild GB structure using updated cuts.
    #     """
    #     sol = self.deep_copy_routes(routes_by_clusters)

    #     # pick cluster with >=2 nodes
    #     cands = [cid] if (cid is not None and cid in sol) else list(sol.keys())
    #     cands = [c for c in cands if len(self._flatten_gbs(sol[c])) >= 2]
    #     if not cands:
    #         return sol
    #     c = int(self.rng.choice(cands))

    #     gbs = sol[c]
    #     flat, cuts = self._cluster_to_flat_and_cuts(gbs)
    #     L = len(flat)
    #     if L < 2:
    #         return sol

    #     # choose remove & insert on the reduced list
    #     i = int(self.rng.integers(0, L))
    #     node = flat.pop(i)
    #     cuts = self._cuts_after_remove(cuts, i)

    #     # insert position in [0, L-1] after pop => length is L-1, insert range [0..L-1]
    #     newL = len(flat)
    #     j = int(self.rng.integers(0, newL + 1))
    #     # avoid trivial no-op as much as possible
    #     if j == i:
    #         j = (j + 1) % (newL + 1)

    #     flat.insert(j, node)
    #     cuts = self._cuts_after_insert(cuts, j)

    #     sol[c] = self._flat_and_cuts_to_cluster(flat, cuts)
    #     return sol


    # def swap_node_intra_route(
    #     self,
    #     routes_by_clusters: RoutesByClusters,
    #     cid: Optional[int] = None,
    # ) -> RoutesByClusters:
    #     """
    #     ONE move: swap two nodes within the SAME route/cluster on the flattened route.
    #     Cuts unchanged (length unchanged), then rebuild by same cuts.
    #     """
    #     sol = self.deep_copy_routes(routes_by_clusters)

    #     cands = [cid] if (cid is not None and cid in sol) else list(sol.keys())
    #     cands = [c for c in cands if len(self._flatten_gbs(sol[c])) >= 2]
    #     if not cands:
    #         return sol
    #     c = int(self.rng.choice(cands))

    #     gbs = sol[c]
    #     flat, cuts = self._cluster_to_flat_and_cuts(gbs)
    #     L = len(flat)
    #     if L < 2:
    #         return sol

    #     i, j = self.rng.choice(L, size=2, replace=False)
    #     i, j = int(i), int(j)
    #     flat[i], flat[j] = flat[j], flat[i]

    #     sol[c] = self._flat_and_cuts_to_cluster(flat, cuts)
    #     return sol


    # def relocate_node_inter_route(
    #     self,
    #     routes_by_clusters: RoutesByClusters,
    #     *,
    #     src_cid: Optional[int] = None,
    #     dst_cid: Optional[int] = None,
    #     protect_min_len: bool = False,  # 保留参数以兼容你现有调用，但此 flat 版不依赖它
    # ) -> RoutesByClusters:
    #     """
    #     ONE move: move 1 node from src route -> dst route on flattened routes,
    #     subject to capacity: load(dst) + demand(node) <= cap.
    #     Rebuild both GB structures using updated cuts.
    #     """
    #     sol = self.deep_copy_routes(routes_by_clusters)

    #     def has_any_node(c: int) -> bool:
    #         return c in sol and len(self._flatten_gbs(sol[c])) > 0

    #     # pick src
    #     if src_cid is not None:
    #         src_cands = [src_cid] if has_any_node(src_cid) else []
    #     else:
    #         src_cands = [c for c in sol.keys() if has_any_node(c)]
    #     if not src_cands:
    #         return sol
    #     src = int(self.rng.choice(src_cands))

    #     src_flat, src_cuts = self._cluster_to_flat_and_cuts(sol[src])
    #     if len(src_flat) == 0:
    #         return sol

    #     # choose a node from src_flat
    #     i = int(self.rng.integers(0, len(src_flat)))
    #     node = src_flat[i]
    #     dem = self.problem.demand[node]

    #     # pick dst with cap feasibility
    #     def cap_ok(dst: int) -> bool:
    #         if dst == src:
    #             return False
    #         return (self._cluster_load(sol[dst]) + dem) <= self.cap

    #     if dst_cid is not None:
    #         dst_cands = [dst_cid] if (dst_cid in sol and cap_ok(dst_cid)) else []
    #     else:
    #         dst_cands = [c for c in sol.keys() if c != src and cap_ok(c)]
    #     if not dst_cands:
    #         return sol
    #     dst = int(self.rng.choice(dst_cands))

    #     dst_flat, dst_cuts = self._cluster_to_flat_and_cuts(sol[dst])

    #     # commit remove in src
    #     node = src_flat.pop(i)
    #     src_cuts = self._cuts_after_remove(src_cuts, i)

    #     # commit insert in dst
    #     j = int(self.rng.integers(0, len(dst_flat) + 1))
    #     dst_flat.insert(j, node)
    #     dst_cuts = self._cuts_after_insert(dst_cuts, j)

    #     # rebuild back to GB structure
    #     sol[src] = self._flat_and_cuts_to_cluster(src_flat, src_cuts)
    #     sol[dst] = self._flat_and_cuts_to_cluster(dst_flat, dst_cuts)
    #     return sol


    # def swap_node_inter_route(
    #     self,
    #     routes_by_clusters: RoutesByClusters,
    #     *,
    #     cid_a: Optional[int] = None,
    #     cid_b: Optional[int] = None,
    # ) -> RoutesByClusters:
    #     """
    #     ONE move: swap 1 node between two different routes on flattened routes,
    #     subject to capacity after swap.
    #     Cuts unchanged (length unchanged), rebuild by same cuts.
    #     """
    #     sol = self.deep_copy_routes(routes_by_clusters)

    #     def has_any_node(c: int) -> bool:
    #         return c in sol and len(self._flatten_gbs(sol[c])) > 0

    #     # pick A
    #     if cid_a is not None:
    #         A_cands = [cid_a] if has_any_node(cid_a) else []
    #     else:
    #         A_cands = [c for c in sol.keys() if has_any_node(c)]
    #     if not A_cands:
    #         return sol
    #     A = int(self.rng.choice(A_cands))

    #     # pick B
    #     if cid_b is not None:
    #         B_cands = [cid_b] if (cid_b != A and has_any_node(cid_b)) else []
    #     else:
    #         B_cands = [c for c in sol.keys() if c != A and has_any_node(c)]
    #     if not B_cands:
    #         return sol
    #     B = int(self.rng.choice(B_cands))

    #     A_flat, A_cuts = self._cluster_to_flat_and_cuts(sol[A])
    #     B_flat, B_cuts = self._cluster_to_flat_and_cuts(sol[B])
    #     if len(A_flat) == 0 or len(B_flat) == 0:
    #         return sol

    #     i = int(self.rng.integers(0, len(A_flat)))
    #     j = int(self.rng.integers(0, len(B_flat)))
    #     nodeA, nodeB = A_flat[i], B_flat[j]
    #     demA = self.problem.demand[nodeA]
    #     demB = self.problem.demand[nodeB]

    #     loadA = self._cluster_load(sol[A])
    #     loadB = self._cluster_load(sol[B])

    #     # cap check after swap
    #     if (loadA - demA + demB > self.cap) or (loadB - demB + demA > self.cap):
    #         return sol

    #     # commit swap
    #     A_flat[i], B_flat[j] = B_flat[j], A_flat[i]

    #     sol[A] = self._flat_and_cuts_to_cluster(A_flat, A_cuts)
    #     sol[B] = self._flat_and_cuts_to_cluster(B_flat, B_cuts)
    #     return sol




   


    # ============================================================
    # Unified dispatcher
    # ============================================================
    def apply(self, op, routes_by_clusters, **kwargs):
        fn = getattr(self, op)

        # 只保留 fn 真正接受的参数
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







