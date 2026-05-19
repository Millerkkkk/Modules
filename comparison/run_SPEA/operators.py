from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import copy
import numpy as np


@dataclass
class DepotRoute:
    start_depot_id: int
    end_depot_id: int
    customers: List[int]


def clone_route(route: DepotRoute) -> DepotRoute:
    return DepotRoute(
        start_depot_id=route.start_depot_id,
        end_depot_id=route.end_depot_id,
        customers=route.customers[:],
    )


def clone_solution(solution: List[DepotRoute]) -> List[DepotRoute]:
    return [clone_route(r) for r in solution]


class BaseOperator:
    def __init__(self, rlsh):
        self.rlsh = rlsh

    # -----------------------------
    # basic helpers
    # -----------------------------
    def route_load_from_customers(self, customers: List[int]) -> float:
        return sum(self.rlsh.demand(cid) for cid in customers)

    def route_load(self, route: DepotRoute) -> float:
        return self.route_load_from_customers(route.customers)

    def refresh_route_metadata(self, route: DepotRoute) -> DepotRoute:
        """
        只刷新 route 的元数据，不改 customers。
        约定：
        - start_depot_id 保持不变
        - end_depot_id 由最后一个客户决定
        """
        if len(route.customers) == 0:
            return route

        route.end_depot_id = self.rlsh.choose_end_depot(
            route.customers,
            route.start_depot_id,
        )
        return route

    def normalize_solution(self, solution: List[DepotRoute]) -> List[DepotRoute]:
        out = []
        for route in solution:
            if len(route.customers) == 0:
                continue
            out.append(self.refresh_route_metadata(route))
        return out

    def select_two_distinct_routes(
        self,
        solution: List[DepotRoute],
        rng: np.random.Generator,
        min_len_each: int = 1,
    ) -> Optional[Tuple[int, int]]:
        idxs = [i for i, r in enumerate(solution) if len(r.customers) >= min_len_each]
        if len(idxs) < 2:
            return None
        a, b = rng.choice(idxs, size=2, replace=False)
        return int(a), int(b)

    def create_single_customer_route(self, customer_id: int) -> DepotRoute:
        start_depot_id = self.rlsh.choose_start_depot(customer_id)
        end_depot_id = self.rlsh.choose_end_depot([customer_id], start_depot_id)
        return DepotRoute(
            start_depot_id=start_depot_id,
            end_depot_id=end_depot_id,
            customers=[customer_id],
        )

    # -----------------------------
    # repair
    # -----------------------------
    def repair(self, solution: List[DepotRoute], rng: np.random.Generator) -> List[DepotRoute]:
        """
        通用修复：
        1. 去重
        2. 补缺
        3. 拆超容量 route
        4. pool 重插
        """
        all_ids = list(self.rlsh.instance.customers)

        seen = set()
        pool: List[int] = []
        cleaned: List[DepotRoute] = []

        # 1) 去重
        for route in solution:
            new_customers = []
            for cid in route.customers:
                if cid not in seen:
                    seen.add(cid)
                    new_customers.append(cid)
                else:
                    pool.append(cid)

            if new_customers:
                cleaned.append(
                    DepotRoute(
                        start_depot_id=route.start_depot_id,
                        end_depot_id=route.end_depot_id,
                        customers=new_customers,
                    )
                )

        # 2) 补缺
        missing = [cid for cid in all_ids if cid not in seen]
        pool.extend(missing)

        # 3) 拆超容量 route
        split_solution: List[DepotRoute] = []
        for route in cleaned:
            current: List[int] = []
            current_load = 0.0

            for cid in route.customers:
                d = self.rlsh.demand(cid)

                if d > self.rlsh.capacity:
                    raise ValueError(f"Customer {cid} demand exceeds vehicle capacity.")

                if current and current_load + d > self.rlsh.capacity:
                    split_solution.append(
                        DepotRoute(
                            start_depot_id=route.start_depot_id,
                            end_depot_id=route.end_depot_id,
                            customers=current,
                        )
                    )
                    current = [cid]
                    current_load = d
                else:
                    current.append(cid)
                    current_load += d

            if current:
                split_solution.append(
                    DepotRoute(
                        start_depot_id=route.start_depot_id,
                        end_depot_id=route.end_depot_id,
                        customers=current,
                    )
                )

        split_solution = self.normalize_solution(split_solution)

        # 4) pool 重插
        repaired = self.rlsh.rebuild_from_pool(split_solution, pool, rng)
        return self.normalize_solution(repaired)
    

class RouteReductionMutation(BaseOperator):
    def choose_route_to_remove(
        self,
        solution: List[DepotRoute],
    ) -> Optional[int]:
        if len(solution) <= 1:
            return None

        # 先删负载最小的 route，比较稳
        best_idx = None
        best_load = float("inf")

        for i, route in enumerate(solution):
            load = self.route_load(route)
            if load < best_load:
                best_load = load
                best_idx = i

        return best_idx

    def apply(
        self,
        solution: List[DepotRoute],
        rng: np.random.Generator,
    ) -> List[DepotRoute]:
        sol = clone_solution(solution)
        idx = self.choose_route_to_remove(sol)
        if idx is None:
            return sol

        pool = sol[idx].customers[:]
        partial = [r for i, r in enumerate(sol) if i != idx]

        rebuilt = self.rlsh.rebuild_from_pool(partial, pool, rng)
        return self.repair(rebuilt, rng) 


class RandomNodeExchangeMutation(BaseOperator):
    def apply(
        self,
        solution: List[DepotRoute],
        rng: np.random.Generator,
    ) -> List[DepotRoute]:
        sol = clone_solution(solution)
        pair = self.select_two_distinct_routes(sol, rng, min_len_each=1)
        if pair is None:
            return sol

        i, j = pair
        r1, r2 = sol[i], sol[j]

        p1 = int(rng.integers(len(r1.customers)))
        p2 = int(rng.integers(len(r2.customers)))

        c1 = r1.customers[p1]
        c2 = r2.customers[p2]

        d1 = self.rlsh.demand(c1)
        d2 = self.rlsh.demand(c2)

        load1 = self.route_load(r1)
        load2 = self.route_load(r2)

        new_load1 = load1 - d1 + d2
        new_load2 = load2 - d2 + d1

        if new_load1 <= self.rlsh.capacity and new_load2 <= self.rlsh.capacity:
            r1.customers[p1], r2.customers[p2] = c2, c1

        return self.repair(sol, rng)


class RandomNodeTransferMutation(BaseOperator):
    def apply(
        self,
        solution: List[DepotRoute],
        rng: np.random.Generator,
    ) -> List[DepotRoute]:
        sol = clone_solution(solution)
        pair = self.select_two_distinct_routes(sol, rng, min_len_each=1)
        if pair is None:
            return sol

        src_idx, dst_idx = pair
        src = sol[src_idx]
        dst = sol[dst_idx]

        pos = int(rng.integers(len(src.customers)))
        cid = src.customers.pop(pos)

        if self.route_load(dst) + self.rlsh.demand(cid) <= self.rlsh.capacity:
            tmp = [clone_route(dst)]
            tmp = self.rlsh.insert_customer_best_position(tmp, cid, rng)
            sol[dst_idx] = tmp[0]
        else:
            src.customers.insert(pos, cid)

        return self.repair(sol, rng)


class RandomArcExchangeMutation(BaseOperator):
    def _random_segment(
        self,
        customers: List[int],
        rng: np.random.Generator,
        min_len: int = 2,
    ) -> Optional[Tuple[int, int]]:
        if len(customers) < min_len:
            return None

        start = int(rng.integers(0, len(customers) - min_len + 1))
        end = int(rng.integers(start + min_len, len(customers) + 1))
        return start, end

    def apply(
        self,
        solution: List[DepotRoute],
        rng: np.random.Generator,
    ) -> List[DepotRoute]:
        sol = clone_solution(solution)
        pair = self.select_two_distinct_routes(sol, rng, min_len_each=2)
        if pair is None:
            return sol

        i, j = pair
        r1, r2 = sol[i], sol[j]

        seg1_idx = self._random_segment(r1.customers, rng, min_len=2)
        seg2_idx = self._random_segment(r2.customers, rng, min_len=2)
        if seg1_idx is None or seg2_idx is None:
            return sol

        s1, e1 = seg1_idx
        s2, e2 = seg2_idx

        seg1 = r1.customers[s1:e1]
        seg2 = r2.customers[s2:e2]

        seg1_d = sum(self.rlsh.demand(c) for c in seg1)
        seg2_d = sum(self.rlsh.demand(c) for c in seg2)

        load1 = self.route_load(r1)
        load2 = self.route_load(r2)

        new_load1 = load1 - seg1_d + seg2_d
        new_load2 = load2 - seg2_d + seg1_d

        if new_load1 <= self.rlsh.capacity and new_load2 <= self.rlsh.capacity:
            r1.customers = r1.customers[:s1] + seg2 + r1.customers[e1:]
            r2.customers = r2.customers[:s2] + seg1 + r2.customers[e2:]

        return self.repair(sol, rng) 


class RandomArcTransferMutation(BaseOperator):
    def _random_segment(
        self,
        customers: List[int],
        rng: np.random.Generator,
        min_len: int = 2,
    ) -> Optional[Tuple[int, int]]:
        if len(customers) < min_len:
            return None

        start = int(rng.integers(0, len(customers) - min_len + 1))
        end = int(rng.integers(start + min_len, len(customers) + 1))
        return start, end

    def apply(
        self,
        solution: List[DepotRoute],
        rng: np.random.Generator,
    ) -> List[DepotRoute]:
        sol = clone_solution(solution)
        pair = self.select_two_distinct_routes(sol, rng, min_len_each=2)
        if pair is None:
            return sol

        src_idx, dst_idx = pair
        src = sol[src_idx]
        dst = sol[dst_idx]

        seg_idx = self._random_segment(src.customers, rng, min_len=2)
        if seg_idx is None:
            return sol

        s, e = seg_idx
        seg = src.customers[s:e]
        seg_d = sum(self.rlsh.demand(c) for c in seg)

        if self.route_load(dst) + seg_d <= self.rlsh.capacity:
            src.customers = src.customers[:s] + src.customers[e:]

            # 逐个插到 dst 的最佳位置
            tmp = [clone_route(dst)]
            for cid in seg:
                tmp = self.rlsh.insert_customer_best_position(tmp, cid, rng)
            sol[dst_idx] = tmp[0]

        return self.repair(sol, rng)


class BaseCrossover(BaseOperator):
    def __init__(self, rlsh, inherit_num: int = 2):
        super().__init__(rlsh)
        self.inherit_num = inherit_num

    def route_customer_set(self, route: DepotRoute) -> set[int]:
        return set(route.customers)

    def choose_non_overlapping_routes(
        self,
        base_routes: List[DepotRoute],
        candidate_routes: List[DepotRoute],
    ) -> List[DepotRoute]:
        occupied = set()
        for route in base_routes:
            occupied.update(route.customers)

        out = []
        for route in candidate_routes:
            if occupied.isdisjoint(route.customers):
                out.append(clone_route(route))
                occupied.update(route.customers)
        return out

    def collect_pool(
        self,
        parent1: List[DepotRoute],
        parent2: List[DepotRoute],
        selected_routes: List[DepotRoute],
    ) -> List[int]:
        selected = set()
        for route in selected_routes:
            selected.update(route.customers)

        universe = set()
        for route in parent1 + parent2:
            universe.update(route.customers)

        return list(universe - selected)

    def finalize_child(
        self,
        partial_child: List[DepotRoute],
        pool: List[int],
        rng: np.random.Generator,
    ) -> List[DepotRoute]:
        rebuilt = self.rlsh.rebuild_from_pool(partial_child, pool, rng)
        return self.repair(rebuilt, rng)


class HeuristicInheritanceCrossover(BaseCrossover):
    def select_inherited_routes(
        self,
        parent: List[DepotRoute],
    ) -> List[DepotRoute]:
        ranked = sorted(parent, key=self.rlsh.route_heuristic_value)
        k = min(self.inherit_num, len(ranked))
        return [clone_route(r) for r in ranked[:k]]

    def apply(
        self,
        parent1: List[DepotRoute],
        parent2: List[DepotRoute],
        rng: np.random.Generator,
    ) -> Tuple[List[DepotRoute], List[DepotRoute]]:
        p1 = clone_solution(parent1)
        p2 = clone_solution(parent2)

        # child1
        inherited1 = self.select_inherited_routes(p1)
        p2_ranked = sorted(p2, key=self.rlsh.route_heuristic_value)
        extra1 = self.choose_non_overlapping_routes(inherited1, p2_ranked)
        partial1 = inherited1 + extra1
        pool1 = self.collect_pool(p1, p2, partial1)
        child1 = self.finalize_child(partial1, pool1, rng)

        # child2
        inherited2 = self.select_inherited_routes(p2)
        p1_ranked = sorted(p1, key=self.rlsh.route_heuristic_value)
        extra2 = self.choose_non_overlapping_routes(inherited2, p1_ranked)
        partial2 = inherited2 + extra2
        pool2 = self.collect_pool(p1, p2, partial2)
        child2 = self.finalize_child(partial2, pool2, rng)

        return child1, child2  


class RandomInheritanceCrossover(BaseCrossover):
    def select_inherited_routes(
        self,
        parent: List[DepotRoute],
        rng: np.random.Generator,
    ) -> List[DepotRoute]:
        if len(parent) == 0:
            return []

        k = min(self.inherit_num, len(parent))
        idxs = rng.choice(len(parent), size=k, replace=False)
        return [clone_route(parent[int(i)]) for i in idxs]

    def apply(
        self,
        parent1: List[DepotRoute],
        parent2: List[DepotRoute],
        rng: np.random.Generator,
    ) -> Tuple[List[DepotRoute], List[DepotRoute]]:
        p1 = clone_solution(parent1)
        p2 = clone_solution(parent2)

        # child1
        inherited1 = self.select_inherited_routes(p1, rng)
        p2_shuffled = clone_solution(p2)
        rng.shuffle(p2_shuffled)
        extra1 = self.choose_non_overlapping_routes(inherited1, p2_shuffled)
        partial1 = inherited1 + extra1
        pool1 = self.collect_pool(p1, p2, partial1)
        child1 = self.finalize_child(partial1, pool1, rng)

        # child2
        inherited2 = self.select_inherited_routes(p2, rng)
        p1_shuffled = clone_solution(p1)
        rng.shuffle(p1_shuffled)
        extra2 = self.choose_non_overlapping_routes(inherited2, p1_shuffled)
        partial2 = inherited2 + extra2
        pool2 = self.collect_pool(p1, p2, partial2)
        child2 = self.finalize_child(partial2, pool2, rng)

        return child1, child2



class MutationManager:
    def __init__(
        self,
        rlsh,
        rrm_times: int = 1,
        rnem_times: int = 10,
        rntm_times: int = 10,
        raem_times: int = 10,
        ratm_times: int = 10,
    ):
        self.plan = (
            [RouteReductionMutation(rlsh)] * rrm_times +
            [RandomNodeExchangeMutation(rlsh)] * rnem_times +
            [RandomNodeTransferMutation(rlsh)] * rntm_times +
            [RandomArcExchangeMutation(rlsh)] * raem_times +
            [RandomArcTransferMutation(rlsh)] * ratm_times
        )

    def apply_all(
        self,
        solution: List[DepotRoute],
        rng: np.random.Generator,
        shuffle: bool = True,
    ) -> List[DepotRoute]:
        # print(solution)
        sol = clone_solution(solution)
        ops = self.plan[:]
        if shuffle:
            rng.shuffle(ops)

        for op in ops:
            sol = op.apply(sol, rng)
        return sol


class CrossoverManager:
    def __init__(
        self,
        rlsh,
        hic_times: int = 2,
        ric_times: int = 2,
        inherit_num: int = 2,
    ):
        self.plan = (
            [HeuristicInheritanceCrossover(rlsh, inherit_num=inherit_num)] * hic_times +
            [RandomInheritanceCrossover(rlsh, inherit_num=inherit_num)] * ric_times
        )

    def apply_one(
        self,
        parent1: List[DepotRoute],
        parent2: List[DepotRoute],
        rng: np.random.Generator,
    ) -> Tuple[List[DepotRoute], List[DepotRoute]]:
        op = self.plan[int(rng.integers(len(self.plan)))]
        return op.apply(parent1, parent2, rng)


