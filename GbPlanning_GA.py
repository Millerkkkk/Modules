from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from scipy.spatial.distance import cdist
from params import GAParams


# =========================
# Params / Result
# =========================



@dataclass
class ACOResult:
    order: List[int]            # GB访问顺序（GB索引）
    cost: float
    history: List[float]


# =========================
# Core: Ant
# =========================

class Ant:
    def __init__(
        self,
        n_nodes: int,
        pheromone: np.ndarray,
        heuristic: np.ndarray,
        rng: np.random.Generator,
    ):
        self.n_nodes = int(n_nodes)
        self.pheromone = pheromone
        self.heuristic = heuristic
        self.rng = rng

        start = int(rng.integers(0, self.n_nodes))
        self.route: List[int] = [start]

        self.visited = {start}
        self.unvisited = [i for i in range(self.n_nodes) if i != start]
        self.cur = start

        self.cost: float = float("inf")

    def select_next(self, alpha: float, beta: float, greedy_prob: float) -> Optional[int]:
        if not self.unvisited:
            return None

        cand = np.array(self.unvisited, dtype=int)

        tau = self.pheromone[self.cur, cand]
        eta = self.heuristic[self.cur, cand]

        score = np.power(tau, alpha) * np.power(eta, beta)
        s = float(score.sum())

        if not np.isfinite(s) or s <= 0:
            # fallback: random
            return int(self.rng.choice(cand))

        prob = score / s

        if self.rng.random() < greedy_prob:
            return int(cand[int(np.argmax(prob))])
        return int(self.rng.choice(cand, p=prob))

    def step(self, nxt: int) -> None:
        self.route.append(nxt)
        self.visited.add(nxt)
        self.unvisited.remove(nxt)
        self.cur = nxt

    def construct(self, alpha: float, beta: float, greedy_prob: float) -> None:
        while self.unvisited:
            nxt = self.select_next(alpha, beta, greedy_prob)
            if nxt is None:
                break
            self.step(nxt)


# =========================
# Core: ACO for GB ordering
# =========================

class GbPlannerGA:
    """
    输入：
      - gb_centers: (m,2)
      - depots_xy:  (d,2)
    输出：
      - GB访问顺序（索引序列）
    """

    def __init__(self, gb_centers: np.ndarray, depots_xy: np.ndarray, params):
        self.params = params
        self.rng = np.random.default_rng(params.seed)

        centers = np.asarray(gb_centers, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError(f"gb_centers must be (m,2), got {centers.shape}")

        depots = np.asarray(depots_xy, dtype=float)
        if depots.ndim != 2 or depots.shape[1] != 2:
            raise ValueError(f"depots_xy must be (d,2), got {depots.shape}")

        self.centers = centers
        self.depots = depots
        self.m = centers.shape[0]

        self.dist = cdist(self.centers, self.centers, metric="euclidean")

        # depot attach precompute：每个GB到最近depot的距离
        # attach[i] = min_depot_dist(GB_i)
        self.attach = cdist(self.centers, self.depots).min(axis=1)

        self.best_route: Optional[np.ndarray] = None
        self.best_cost: float = float("inf")

    def evaluate(self, route: np.ndarray) -> float:
        """route: shape (m,), permutation"""
        if route.size == 0:
            return float("inf")
        # 内部距离和
        total = float(self.dist[route[:-1], route[1:]].sum())
        # depot 首尾连接（最近 depot）
        total += float(self.attach[route[0]] + self.attach[route[-1]])
        return total

    # ---------- GA Operators ----------
    def init_population(self) -> np.ndarray:
        """Return population as (pop_size, m) permutations."""
        pop = np.empty((self.params.pop_size, self.m), dtype=int)
        base = np.arange(self.m, dtype=int)
        for i in range(self.params.pop_size):
            pop[i] = self.rng.permutation(base)
        return pop

    def tournament_select(self, pop: np.ndarray, costs: np.ndarray) -> np.ndarray:
        """Select one parent by tournament (min cost wins)."""
        k = self.params.tournament_k
        idx = self.rng.integers(0, pop.shape[0], size=k)
        best = idx[np.argmin(costs[idx])]
        return pop[best].copy()

    def ox_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Order Crossover (OX) for permutations."""
        n = p1.size
        a, b = sorted(self.rng.choice(n, size=2, replace=False))
        child = np.full(n, -1, dtype=int)
        # copy slice from p1
        child[a:b+1] = p1[a:b+1]
        # fill remaining from p2 in order
        fill = [x for x in p2 if x not in child]
        ptr = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill[ptr]
                ptr += 1
        return child

    def mutate(self, x: np.ndarray) -> None:
        """In-place mutation: inversion or swap."""
        n = x.size
        if n <= 2:
            return
        if self.rng.random() < self.params.inversion_rate:
            i, j = sorted(self.rng.choice(n, size=2, replace=False))
            x[i:j+1] = x[i:j+1][::-1]
        else:
            i, j = self.rng.choice(n, size=2, replace=False)
            x[i], x[j] = x[j], x[i]

    def solve(self) -> ACOResult:
        history: List[float] = []

        pop = self.init_population()
        costs = np.array([self.evaluate(ind) for ind in pop], dtype=float)

        # 初始化全局最优
        best_idx = int(np.argmin(costs))
        self.best_route = pop[best_idx].copy()
        self.best_cost = float(costs[best_idx])
        history.append(self.best_cost)

        for _gen in range(self.params.num_gen):
            # 精英保留
            elite_size = max(0, int(self.params.elite_size))
            elite_idx = np.argsort(costs)[:elite_size]
            elites = pop[elite_idx].copy()
            elites_cost = costs[elite_idx].copy()

            # 生成新一代
            new_pop = []
            if elite_size > 0:
                for e in elites:
                    new_pop.append(e)

            while len(new_pop) < self.params.pop_size:
                p1 = self.tournament_select(pop, costs)
                p2 = self.tournament_select(pop, costs)

                if self.rng.random() < self.params.crossover_rate:
                    c1 = self.ox_crossover(p1, p2)
                    c2 = self.ox_crossover(p2, p1)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                if self.rng.random() < self.params.mutation_rate:
                    self.mutate(c1)
                if self.rng.random() < self.params.mutation_rate:
                    self.mutate(c2)

                new_pop.append(c1)
                if len(new_pop) < self.params.pop_size:
                    new_pop.append(c2)

            pop = np.array(new_pop, dtype=int)
            costs = np.array([self.evaluate(ind) for ind in pop], dtype=float)

            # 更新全局最优
            gen_best_idx = int(np.argmin(costs))
            gen_best_cost = float(costs[gen_best_idx])
            if gen_best_cost < self.best_cost:
                self.best_cost = gen_best_cost
                self.best_route = pop[gen_best_idx].copy()

            history.append(self.best_cost)

        return ACOResult(
            order=(self.best_route.tolist() if self.best_route is not None else []),
            cost=self.best_cost,
            history=history,
        )

# =========================
# Simple functional API
# =========================

def plan_gb_order_ga(
    gb_centers: np.ndarray,
    depots_xy: np.ndarray,
    params: Optional[GAParams] = None,
) -> ACOResult:
    params = params or GAParams()
    return GbPlannerGA(gb_centers, depots_xy, params).solve()
