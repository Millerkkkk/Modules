from __future__ import annotations
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math
import numpy as np

from EVRP.loader import load_problem
from evaluate import Evaluator
from params import ev_params
from stochastic_demand import generate_normal_scenarios


# =========================================================
# Basic data structures
# =========================================================

@dataclass
class Individual:
    solution: object
    objectives: Optional[List[float]] = None

    strength: float = 0.0
    raw_fitness: float = 0.0
    density: float = 0.0
    fitness: float = 0.0

    def copy(self) -> "Individual":
        return Individual(
            solution=self.solution,
            objectives=None if self.objectives is None else self.objectives[:],
            strength=self.strength,
            raw_fitness=self.raw_fitness,
            density=self.density,
            fitness=self.fitness,
        )
    

@dataclass
class Archive:
    individuals: List[Individual] = field(default_factory=list)

    def clear(self) -> None:
        self.individuals.clear()

    def size(self) -> int:
        return len(self.individuals)

    def copy(self) -> "Archive":
        return Archive(individuals=[ind.copy() for ind in self.individuals])



# =========================================================
# Utilities
# =========================================================

def clone_solution(solution):
    # 假设你的 solution 是 List[DepotRoute]
    # 每个 route 有 start_depot_id / end_depot_id / customers
    return [
        type(r)(
            start_depot_id=r.start_depot_id,
            end_depot_id=r.end_depot_id,
            customers=r.customers[:],
        )
        for r in solution
    ]

def dominates(a: List[float], b: List[float]) -> bool:
    """
    最小化问题的 Pareto dominance
    a dominates b iff:
      - a 在所有目标上都不差于 b
      - 且至少一个目标上严格更好
    """
    no_worse = all(x <= y for x, y in zip(a, b))
    strictly_better = any(x < y for x, y in zip(a, b))
    return no_worse and strictly_better


def euclidean_distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def unique_by_objectives(individuals):
    unique = []
    seen = set()

    for ind in individuals:
        key = tuple(ind.objectives)
        if key not in seen:
            seen.add(key)
            unique.append(ind)

    return unique


def get_nondominated(individuals):
    result = []
    for i, ind_i in enumerate(individuals):
        dominated_flag = False
        for j, ind_j in enumerate(individuals):
            if i == j:
                continue
            if dominates(ind_j.objectives, ind_i.objectives):
                dominated_flag = True
                break
        if not dominated_flag:
            result.append(ind_i)
    return result


# =========================================================
# SPEA Optimizer
# =========================================================

class SPEAOptimizer:
    def __init__(
        self,
        initializer,
        evaluator,
        crossover_manager,
        mutation_manager,
        scenarios,
        pop_size: int,
        archive_size: int,
        max_generations: int,
        rng: np.random.Generator,
        crossover_prob: float = 1.0,
        mutation_prob: float = 1.0,
        tournament_size: int = 2,
        k_neighbor: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        参数说明
        ----------
        initializer:
            需要有 initialize_population(pop_size, rng) -> List[solution]

        evaluator:
            需要有 evaluate_given_scenarios(solution, scenarios=None)
            或 evaluate_saa(solution, ...)
            返回对象里至少有:
              - total_cost
              - environmental_cost
              - total_satisfaction

        crossover_manager:
            需要有 apply_one(parent1, parent2, rng) -> (child1, child2)

        mutation_manager:
            需要有 apply_all(solution, rng) -> solution

        注意：
        ----------
        这里统一把目标写成最小化：
          [total_cost, environmental_cost, -total_satisfaction]
        """
        self.initializer = initializer
        self.evaluator = evaluator
        self.crossover_manager = crossover_manager
        self.mutation_manager = mutation_manager
        self.scenarios = scenarios

        self.pop_size = pop_size
        self.archive_size = archive_size
        self.max_generations = max_generations
        self.rng = rng

        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.k_neighbor = k_neighbor
        self.verbose = verbose

    # -----------------------------------------------------
    # evaluation
    # -----------------------------------------------------
    
    def evaluate_solution(self, solution):
        routes = {}

        for i, s in enumerate(solution):
            route = [s.start_depot_id] + list(s.customers) + [s.end_depot_id]
            routes[i] = route


        result = self.evaluator.evaluate_saa(routes, self.scenarios)

        total_cost = float(result["total_cost"])
        total_ra = float(result["total_ra"])
        return [
            float(total_cost),
            float(total_ra)
        ]

    def evaluate_population(self, population: List[Individual]) -> None:
        for ind in population:
            if ind.objectives is None:
                ind.objectives = self.evaluate_solution(ind.solution)

    # -----------------------------------------------------
    # fitness assignment (SPEA2 style)
    # -----------------------------------------------------

    def strength_assignment(self, union: List[Individual]) -> None:
        n = len(union)

        for i in range(n):
            s = 0
            for j in range(n):
                if i == j:
                    continue
                if dominates(union[i].objectives, union[j].objectives):
                    s += 1
            union[i].strength = float(s)

        for i in range(n):
            raw = 0.0
            for j in range(n):
                if i == j:
                    continue
                if dominates(union[j].objectives, union[i].objectives):
                    raw += union[j].strength
            union[i].raw_fitness = raw

    def density_estimation(self, union: List[Individual]) -> None:
        n = len(union)
        if n == 0:
            return

        # 常用设置: k = sqrt(N)
        k = self.k_neighbor
        if k is None:
            k = max(1, int(math.sqrt(n)))
        k = min(k, max(1, n - 1))

        distance_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = euclidean_distance(union[i].objectives, union[j].objectives)
                distance_matrix[i][j] = d
                distance_matrix[j][i] = d

        for i in range(n):
            distances = [distance_matrix[i][j] for j in range(n) if j != i]
            distances.sort()
            sigma_k = distances[k - 1] if distances else 0.0
            union[i].density = 1.0 / (sigma_k + 2.0)

        for ind in union:
            ind.fitness = ind.raw_fitness + ind.density

    def assign_fitness(self, union: List[Individual]) -> None:
        self.strength_assignment(union)
        self.density_estimation(union)

    # -----------------------------------------------------
    # archive selection
    # -----------------------------------------------------

    def environmental_selection(self, union: List[Individual]) -> Archive:
        """
        SPEA2 标准思路：
        1. raw_fitness < 1 的先进 archive
        2. 不足则按 fitness 最小补足
        3. 超出则做 truncation
        """
        archive_candidates = [ind.copy() for ind in union if ind.raw_fitness < 1.0]
        archive_candidates = unique_by_objectives(archive_candidates)

        if len(archive_candidates) < self.archive_size:
            remaining = [ind.copy() for ind in union if ind.raw_fitness >= 1.0]
            remaining.sort(key=lambda x: x.fitness)
            remaining = unique_by_objectives(remaining)

            need = self.archive_size - len(archive_candidates)
            archive_candidates.extend(remaining[:need])
            return Archive(archive_candidates)

        if len(archive_candidates) == self.archive_size:
            return Archive(archive_candidates)

        truncated = self.truncate_archive(archive_candidates, self.archive_size)
        return Archive(truncated)

    def truncate_archive(self, archive_inds: List[Individual], target_size: int) -> List[Individual]:
        """
        距离截断：
        反复删除“与别人过近”的个体，保持分散性
        """
        inds = archive_inds[:]

        while len(inds) > target_size:
            n = len(inds)

            # 计算两两距离
            distance_lists = []
            for i in range(n):
                distances = []
                for j in range(n):
                    if i == j:
                        continue
                    d = euclidean_distance(inds[i].objectives, inds[j].objectives)
                    distances.append(d)
                distances.sort()
                distance_lists.append(distances)

            # 选最拥挤的个体删除
            remove_idx = None
            best_signature = None

            for i, dlist in enumerate(distance_lists):
                signature = tuple(dlist)
                if best_signature is None or signature < best_signature:
                    best_signature = signature
                    remove_idx = i

            inds.pop(remove_idx)

        return inds

    # -----------------------------------------------------
    # selection
    # -----------------------------------------------------

    def tournament_select_one(self, archive: Archive) -> Individual:
        """
        二元锦标赛：
        fitness 小者优先
        """
        if archive.size() == 1:
            return archive.individuals[0]

        idxs = self.rng.choice(len(archive.individuals), size=self.tournament_size, replace=False)
        candidates = [archive.individuals[int(i)] for i in idxs]
        candidates.sort(key=lambda ind: ind.fitness)
        return candidates[0]

    def mating_selection(self, archive: Archive) -> List[Individual]:
        mating_pool = []
        while len(mating_pool) < self.pop_size:
            mating_pool.append(self.tournament_select_one(archive))
        return mating_pool

    # -----------------------------------------------------
    # variation
    # -----------------------------------------------------

    def crossover(self, parent1, parent2):
        if self.rng.random() <= self.crossover_prob:
            c1, c2 = self.crossover_manager.apply_one(parent1, parent2, self.rng)
            return c1, c2
        return clone_solution(parent1), clone_solution(parent2)

    def mutate(self, solution):
        if self.rng.random() <= self.mutation_prob:
            return self.mutation_manager.apply_all(solution, self.rng)
        return clone_solution(solution)

    def reproduce(self, mating_pool: List[Individual]) -> List[Individual]:
        offspring: List[Individual] = []

        order = list(range(len(mating_pool)))
        self.rng.shuffle(order)

        for k in range(0, len(order), 2):
            p1 = mating_pool[order[k]].solution
            if k + 1 < len(order):
                p2 = mating_pool[order[k + 1]].solution
            else:
                p2 = mating_pool[order[0]].solution

            child_sol1, child_sol2 = self.crossover(p1, p2)
            child_sol1 = self.mutate(child_sol1)
            child_sol2 = self.mutate(child_sol2)

            offspring.append(Individual(solution=child_sol1))
            if len(offspring) < self.pop_size:
                offspring.append(Individual(solution=child_sol2))

            if len(offspring) >= self.pop_size:
                break

        return offspring[:self.pop_size]

    # -----------------------------------------------------
    # initialization
    # -----------------------------------------------------

    def initialize_population(self) -> List[Individual]:
        population = self.initializer.initialize_population(self.rng)
        population = [Individual(solution=sol) for sol in population]
        self.evaluate_population(population)
        return population

    # -----------------------------------------------------
    # one generation
    # -----------------------------------------------------

    def step(self, population: List[Individual], archive: Archive, generation: int):
        # union = population + archive
        union = [ind.copy() for ind in population] + [ind.copy() for ind in archive.individuals]

        # fitness assignment
        self.assign_fitness(union)

        # archive update
        archive = self.environmental_selection(union)

        # mating selection from archive
        mating_pool = self.mating_selection(archive)

        # offspring generation
        offspring = self.reproduce(mating_pool)

        # evaluate offspring
        self.evaluate_population(offspring)

        if self.verbose:
            objs = [ind.objectives for ind in archive.individuals]
            best_f1 = min(obj[0] for obj in objs) if objs else float("nan")
            best_f2 = min(obj[1] for obj in objs) if objs else float("nan")
            # print(
            #     f"[Gen {generation:03d}] "
            #     f"archive={archive.size():3d} "
            #     f"best(cost)={best_f1:.4f} "
            #     f"best(RA)={best_f2:.4f} "
            # )

        return offspring, archive

    # -----------------------------------------------------
    # run
    # -----------------------------------------------------

    def run(self) -> Archive:
        population = self.initialize_population()
        archive = Archive([])

        for gen in range(1, self.max_generations + 1):
            population, archive = self.step(population, archive, gen)

        # 最后一轮再基于最终 union 更新一次 archive，避免遗漏
        union = [ind.copy() for ind in population] + [ind.copy() for ind in archive.individuals]
        self.assign_fitness(union)
        archive = self.environmental_selection(union)

        # 最终只保留非支配解
        final_nd = get_nondominated(archive.individuals)

        if self.verbose:
            print(f"Done. Final archive size = {archive.size()}")
            print(f"Final nondominated size = {len(final_nd)}")

        return Archive(final_nd)


