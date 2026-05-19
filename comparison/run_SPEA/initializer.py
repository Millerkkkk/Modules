from __future__ import annotations
from dataclasses import dataclass
import os
from typing import List
import numpy as np

from solution import Route, RouteSolution

@dataclass
class DepotRoute:
    start_depot_id: int
    end_depot_id: int
    customers: List[int]


@dataclass
class InitialPopulationConfig:
    population_size: int
    heuristic_ratio: float = 0.3   # 论文实验里 30% 表现更好
    shuffle_customers_before_rlsh: bool = True
    randomize_rlsh_lambda: bool = False
    lambda_low: float = 0.2
    lambda_high: float = 0.8


class HybridInitializer:
    """
    论文 Initial Population 思路的适配版：
    - heuristic 部分：用 RLSH 构造
    - random 部分：只按容量随机装 route，不考虑距离
    - 最后统一 repair 一次

    注意：
    论文原文是 single-depot；这里是你当前 multi-depot 版本的工程化适配。
    """

    def __init__(self, instance, rlsh_constructor, config: InitialPopulationConfig):
        self.instance = instance
        self.rlsh = rlsh_constructor
        self.config = config

    # ---------------------------------------------------------
    # basic helpers
    # ---------------------------------------------------------

    @property
    def capacity(self) -> float:
        return float(self.instance.vehicle_capacity)

    @property
    def depot_ids(self) -> List[int]:
        return list(self.instance.depots)

    def customer_ids(self) -> List[int]:
        return list(self.instance.customers)

    def demand(self, customer_id: int) -> float:
        return float(self.instance.demand[customer_id])

    def nearest_depot(self, node_id: int) -> int:
        """
        这里假设 instance.distance_matrix 是 ndarray，可按 [i, j] 访问。
        如果你封装了 instance.distance(i, j)，把这里改掉即可。
        """
        best_depot = None
        best_dist = float("inf")

        for depot_id in self.depot_ids:
            d = float(self.instance.distance_matrix[node_id, depot_id])
            if d < best_dist:
                best_dist = d
                best_depot = depot_id

        return int(best_depot)

    def choose_start_depot_for_route(self, first_customer: int) -> int:
        return self.nearest_depot(first_customer)

    def choose_end_depot_for_route(self, last_customer: int, start_depot_id: int) -> int:
        return self.nearest_depot(last_customer) if last_customer is not None else start_depot_id

    # ---------------------------------------------------------
    # heuristic individuals
    # ---------------------------------------------------------

    def build_heuristic_individual(self, rng: np.random.Generator) -> List[DepotRoute]:
        customer_ids = self.customer_ids()

        if self.config.shuffle_customers_before_rlsh:
            customer_ids = customer_ids[:]
            rng.shuffle(customer_ids)

        # 可选：给不同个体不同 lambda，增强多样性
        old_lam = getattr(self.rlsh, "lam", None)
        if self.config.randomize_rlsh_lambda and old_lam is not None:
            self.rlsh.lam = float(rng.uniform(self.config.lambda_low, self.config.lambda_high))

        solution = self.rlsh.construct_solution(customer_ids, rng)
        # solution = self.rlsh.repair_solution(solution, rng)

        if self.config.randomize_rlsh_lambda and old_lam is not None:
            self.rlsh.lam = old_lam

        return solution

    # ---------------------------------------------------------
    # random individuals
    # ---------------------------------------------------------

    def random_customer_order(self, rng: np.random.Generator) -> List[int]:
        ids = self.customer_ids()
        rng.shuffle(ids)
        return ids

    def split_by_capacity_random_only(
        self,
        ordered_customers: List[int],
        rng: np.random.Generator,
    ) -> List[DepotRoute]:
        """
        论文里 random 部分“只基于容量，忽略距离计算”。这里对应实现为：
        - 随机排列客户
        - 顺次装车，超容量就开新 route
        - route 的 depot 用简单规则补上：
          start = 第一位客户最近 depot
          end   = 最后一位客户最近 depot
        """
        solution: List[DepotRoute] = []

        current_route: List[int] = []
        current_load = 0.0

        for cid in ordered_customers:
            d = self.demand(cid)

            # 如果存在单客户需求超过容量，这里不直接 silent 地吞掉
            if d > self.capacity:
                raise ValueError(
                    f"Customer {cid} demand ({d}) exceeds vehicle capacity ({self.capacity})."
                )

            if current_route and current_load + d > self.capacity:
                start_depot_id = self.choose_start_depot_for_route(current_route[0])
                end_depot_id = self.choose_end_depot_for_route(current_route[-1], start_depot_id)
                solution.append(
                    DepotRoute(
                        start_depot_id=start_depot_id,
                        end_depot_id=end_depot_id,
                        customers=current_route[:],
                    )
                )
                current_route = [cid]
                current_load = d
            else:
                current_route.append(cid)
                current_load += d

        if current_route:
            start_depot_id = self.choose_start_depot_for_route(current_route[0])
            end_depot_id = self.choose_end_depot_for_route(current_route[-1], start_depot_id)
            solution.append(
                DepotRoute(
                    start_depot_id=start_depot_id,
                    end_depot_id=end_depot_id,
                    customers=current_route[:],
                )
            )

        return solution

    def build_random_individual(self, rng: np.random.Generator) -> List[DepotRoute]:
        ordered_customers = self.random_customer_order(rng)
        solution = self.split_by_capacity_random_only(ordered_customers, rng)
        # solution = self.rlsh.repair_solution(solution, rng)
        return solution

    # ---------------------------------------------------------
    # population generation
    # ---------------------------------------------------------

    def initialize_population(self, rng: np.random.Generator) -> List[List[DepotRoute]]:
        pop_size = self.config.population_size
        heuristic_count = int(round(pop_size * self.config.heuristic_ratio))
        heuristic_count = max(0, min(pop_size, heuristic_count))
        random_count = pop_size - heuristic_count

        population: List[List[DepotRoute]] = []

        for _ in range(heuristic_count):
            population.append(self.build_heuristic_individual(rng))

        for _ in range(random_count):
            population.append(self.build_random_individual(rng))

        # 再打乱一次，避免前半段全是 heuristic
        rng.shuffle(population)
        return population
    




    