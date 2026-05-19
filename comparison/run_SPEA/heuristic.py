from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class DepotRoute:
    start_depot_id: int
    end_depot_id: int
    customers: List[int]


@dataclass
class HeuristicScore:
    customer_id: int
    distance_term: float
    demand_remainder_term: float
    score: float


class RLSHConstructor:
    """
    Resultant Local Search Heuristic (RLSH)

    论文核心思想：
    HR_i = lambda * d(current, i) + (1-lambda) * DR_i

    这里的实现采用：
    DR_i = remaining_capacity - demand_i

    说明：
    - 论文文字对 DR_i 的定义略有歧义；
    - 为了让“路线中段”也合理，使用 remaining_capacity 更自然。
    """
    def __init__(
        self,
        instance,
        lam: float = 0.7,
        normalize_terms: bool = True,
        infeasible_penalty: float = 1e9,
    ):
        self.instance = instance
        self.lam = lam
        self.normalize_terms = normalize_terms
        self.infeasible_penalty = infeasible_penalty

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    @property
    def depot_ids(self) -> List[int]:
        return list(self.instance.depots)
    
    @property
    def capacity(self) -> float:
        return self.instance.vehicle_capacity
    
    def demand(self, customer_id: int) -> float:
        return float(self.instance.demand[customer_id])
    
    def distance(self, i: int, j: int) -> float:
        return float(self.instance.distance_matrix[i, j])

    def route_load(self, route) -> float:
        return sum(self.demand(cid) for cid in route)

    def remaining_capacity(self, route) -> float:
        return self.capacity - self.route_load(route)
    

    # ------------------------------------------------------------------
    # Heuristic terms
    # ------------------------------------------------------------------

    def demand_remainder(self, remaining_capacity: float, customer_id: int) -> float:
        """
        DR_i: 剩余容量与客户需求的差。
        越接近 0 越好；<0 表示不可行。
        """
        return remaining_capacity - self.demand(customer_id)

    def _normalize_values(self, values: List[float]) -> List[float]:
        if not values:
            return values
        vmin = min(values)
        vmax = max(values)
        if abs(vmax - vmin) < 1e-12:
            return [0.0 for _ in values]
        return [(v - vmin) / (vmax - vmin) for v in values]
    
    def nearest_depot(self, node_id: int) -> int:
        best_depot = None
        best_dist = float("inf")

        for depot_id in self.depot_ids:
            d = self.distance(node_id, depot_id)
            if d < best_dist:
                best_dist = d
                best_depot = depot_id

        return best_depot

    def score_candidates(
        self,
        current_node: int,
        candidate_ids: List[int],
        remaining_capacity: float,
    ) -> List[tuple[int, float]]:
        if not candidate_ids:
            return []

        raw_dist = [self.distance(current_node, cid) for cid in candidate_ids]
        raw_dr = [abs(self.demand_remainder(remaining_capacity, cid)) for cid in candidate_ids]

        if self.normalize_terms:
            dist_terms = self._normalize_values(raw_dist)
            dr_terms = self._normalize_values(raw_dr)
        else:
            dist_terms = raw_dist
            dr_terms = raw_dr

        scores = []
        for cid, d_term, dr_term in zip(candidate_ids, dist_terms, dr_terms):
            score = self.lam * d_term + (1.0 - self.lam) * dr_term
            scores.append((cid, score))

        scores.sort(key=lambda x: (x[1], x[0]))
        return scores
    
    def select_next_customer(
        self,
        current_node: int,
        candidate_ids: List[int],
        remaining_capacity: float,
        rng: np.random.Generator,
    ) -> Optional[int]:
        feasible = [cid for cid in candidate_ids if self.demand(cid) <= remaining_capacity]
        if not feasible:
            return None

        scores = self.score_candidates(current_node, feasible, remaining_capacity)
        if not scores:
            return None

        best_score = scores[0][1]
        best = [cid for cid, score in scores if abs(score - best_score) < 1e-12]
        return int(rng.choice(best))

    def choose_start_depot(self, customer_id: int) -> int:
        return self.nearest_depot(customer_id)
    
    def choose_end_depot(self, route: List[int], start_depot_id: int) -> int:
        if not route:
            return start_depot_id
        last_customer = route[-1]
        return self.nearest_depot(last_customer)
    
    def remove_empty_routes(self, solution: List[DepotRoute]) -> List[DepotRoute]:
        return [r for r in solution if len(r.customers) > 0]
    
    # ------------------------------------------------------------------
    # Route construction
    # ------------------------------------------------------------------

    def construct_route(
        self,
        start_depot_id: int,
        unrouted_ids: List[int],
        rng: np.random.Generator,
    ) -> List[int]:
        if not unrouted_ids:
            return []

        route = []
        current_node = start_depot_id
        remaining = unrouted_ids[:]

        while True:
            rem_cap = self.remaining_capacity(route)
            next_customer = self.select_next_customer(
                current_node=current_node,
                candidate_ids=remaining,
                remaining_capacity=rem_cap,
                rng=rng,
            )

            if next_customer is None:
                break

            route.append(next_customer)
            remaining.remove(next_customer)
            current_node = next_customer

        return route
    

    def construct_solution(
        self,
        customer_ids: List[int],
        rng: np.random.Generator,
    ) -> List[DepotRoute]:
        unrouted = customer_ids[:]
        solution: List[DepotRoute] = []

        while unrouted:
            seed_customer = unrouted[0]
            start_depot_id = self.choose_start_depot(seed_customer)

            route_customers = self.construct_route(
                start_depot_id=start_depot_id,
                unrouted_ids=unrouted,
                rng=rng,
            )

            if len(route_customers) == 0:
                route_customers = [seed_customer]

            end_depot_id = self.choose_end_depot(route_customers, start_depot_id)

            solution.append(
                DepotRoute(
                    start_depot_id=start_depot_id,
                    end_depot_id=end_depot_id,
                    customers=route_customers,
                )
            )

            routed_set = set(route_customers)
            unrouted = [cid for cid in unrouted if cid not in routed_set]

        return self.remove_empty_routes(solution)
    
    # ---------------------------------------------------------
    # insertion / rebuild / repair
    # ---------------------------------------------------------

    def insertion_cost_proxy(
        self,
        start_depot_id: int,
        end_depot_id: int,
        route: List[int],
        pos: int,
        customer_id: int,
    ) -> float:
        prev_node = start_depot_id if pos == 0 else route[pos - 1]
        next_node = end_depot_id if pos == len(route) else route[pos]

        old_cost = self.distance(prev_node, next_node)
        new_cost = self.distance(prev_node, customer_id) + self.distance(customer_id, next_node)
        return new_cost - old_cost

    def best_insertion_position(
        self,
        start_depot_id: int,
        end_depot_id: int,
        route: List[int],
        customer_id: int,
    ) -> Optional[int]:
        if self.route_load(route) + self.demand(customer_id) > self.capacity:
            return None

        best_pos = None
        best_cost = float("inf")

        for pos in range(len(route) + 1):
            c = self.insertion_cost_proxy(start_depot_id, end_depot_id, route, pos, customer_id)
            if c < best_cost:
                best_cost = c
                best_pos = pos

        return best_pos

    def insert_customer_best_position(
        self,
        solution: List[DepotRoute],
        customer_id: int,
        rng: np.random.Generator,
    ) -> List[DepotRoute]:
        best_route_idx = None
        best_pos = None
        best_cost = float("inf")

        for ridx, depot_route in enumerate(solution):
            pos = self.best_insertion_position(
                depot_route.start_depot_id,
                depot_route.end_depot_id,
                depot_route.customers,
                customer_id,
            )
            if pos is None:
                continue

            c = self.insertion_cost_proxy(
                depot_route.start_depot_id,
                depot_route.end_depot_id,
                depot_route.customers,
                pos,
                customer_id,
            )
            if c < best_cost:
                best_cost = c
                best_route_idx = ridx
                best_pos = pos

        if best_route_idx is None:
            start_depot_id = self.choose_start_depot(customer_id)
            end_depot_id = self.nearest_depot(customer_id)
            solution.append(
                DepotRoute(
                    start_depot_id=start_depot_id,
                    end_depot_id=end_depot_id,
                    customers=[customer_id],
                )
            )
            return solution

        solution[best_route_idx].customers.insert(best_pos, customer_id)
        solution[best_route_idx].end_depot_id = self.choose_end_depot(
            solution[best_route_idx].customers,
            solution[best_route_idx].start_depot_id,
        )
        return solution

    def rebuild_from_pool(
        self,
        partial_solution: List[DepotRoute],
        relocation_pool: List[int],
        rng: np.random.Generator,
    ) -> List[DepotRoute]:
        solution = [
            DepotRoute(
                start_depot_id=r.start_depot_id,
                end_depot_id=r.end_depot_id,
                customers=r.customers[:],
            )
            for r in partial_solution
        ]

        pool = sorted(relocation_pool, key=lambda cid: self.demand(cid), reverse=True)

        for cid in pool:
            solution = self.insert_customer_best_position(solution, cid, rng)

        return self.remove_empty_routes(solution)

    def route_heuristic_value(self, depot_route: DepotRoute) -> float:
        route = depot_route.customers
        if not route:
            return float("inf")

        current_node = depot_route.start_depot_id
        rem_cap = self.capacity
        vals = []

        for cid in route:
            d = self.distance(current_node, cid)
            dr = abs(rem_cap - self.demand(cid))
            vals.append(self.lam * d + (1.0 - self.lam) * dr)
            rem_cap -= self.demand(cid)
            current_node = cid

        vals.append(self.distance(current_node, depot_route.end_depot_id))
        return float(np.mean(vals))

    def sort_routes_by_heuristic(self, solution: List[DepotRoute]) -> List[DepotRoute]:
        return sorted(solution, key=self.route_heuristic_value)
    
