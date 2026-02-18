from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from scipy.spatial.distance import cdist
from params import ACOParams


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

class GbPlanner:
    """
    输入：
      - gb_centers: (m,2) 每个GB中心
      - depots_xy: (d,2) depot坐标
    输出：
      - GB访问顺序（索引序列）
    """

    def __init__(self, gb_centers: np.ndarray, depots_xy: np.ndarray, params: ACOParams):
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

        # 距离矩阵 / 启发式矩阵
        self.dist = cdist(self.centers, self.centers, metric="euclidean")
        self.heur = 1.0 / (self.dist + 1e-6)
        np.fill_diagonal(self.heur, 0.0)

        # 信息素矩阵
        self.tau = np.ones((self.m, self.m), dtype=float)
        np.fill_diagonal(self.tau, 0.0)

        # 全局最优
        self.best_route: Optional[List[int]] = None
        self.best_cost: float = float("inf")

    def _depot_attach_cost(self, first: int, last: int) -> float:
        # depot -> first / last -> depot 取最小
        first_xy = self.centers[first].reshape(1, -1)
        last_xy = self.centers[last].reshape(1, -1)

        d1 = float(cdist(first_xy, self.depots)[0].min())
        d2 = float(cdist(last_xy, self.depots)[0].min())
        return d1 + d2

    def evaluate(self, route: List[int]) -> float:
        if not route:
            return float("inf")

        # 内部距离
        total = 0.0
        for i in range(len(route) - 1):
            total += self.dist[route[i], route[i + 1]]

        # 加 depot 首尾连接
        total += self._depot_attach_cost(route[0], route[-1])
        return float(total)

    def update_pheromone(self, ants: List[Ant], elite_route: List[int], elite_cost: float) -> None:
        rho = self.params.rho
        eps = self.params.epsilon

        delta = np.zeros_like(self.tau)

        for ant in ants:
            cost = ant.cost
            if not np.isfinite(cost) or cost <= 0:
                continue
            r = ant.route
            for i in range(len(r) - 1):
                delta[r[i], r[i + 1]] += 1.0 / cost

        elite_delta = np.zeros_like(self.tau)
        elite_cost = elite_cost if elite_cost > 0 else 1e-6
        for i in range(len(elite_route) - 1):
            elite_delta[elite_route[i], elite_route[i + 1]] += 1.0 / elite_cost

        self.tau = (1 - rho) * self.tau + delta + eps * elite_delta
        np.fill_diagonal(self.tau, 0.0)

    def solve(self) -> ACOResult:
        history: List[float] = []

        for _ in range(self.params.num_iter):
            ants: List[Ant] = []

            for _k in range(self.params.num_ants):
                ant = Ant(self.m, self.tau, self.heur, self.rng)
                ant.construct(self.params.alpha, self.params.beta, self.params.greedy_prob)
                ant.cost = self.evaluate(ant.route)
                ants.append(ant)

                if ant.cost < self.best_cost:
                    self.best_cost = ant.cost
                    self.best_route = list(ant.route)

            # 迭代结束更新信息素（用当前全局最优做 elite）
            if self.best_route is None:
                continue
            self.update_pheromone(ants, self.best_route, self.best_cost)

            history.append(self.best_cost)

        return ACOResult(order=self.best_route or [], cost=self.best_cost, history=history)


# =========================
# Simple functional API
# =========================

def plan_gb_order(
    gb_centers: np.ndarray,
    depots_xy: np.ndarray,
    params: Optional[ACOParams] = None,
) -> ACOResult:
    params = params or ACOParams()
    return GbPlanner(gb_centers, depots_xy, params).solve()
