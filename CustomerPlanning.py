from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal
import numpy as np

from params import ACOParams



# =========================
# Params / Context / Result
# =========================

DepotMode = Literal["depart", "return", "depart & return"]



@dataclass(frozen=True)
class RouteContext:
    depot_mode: Optional[DepotMode] = None
    prev_last_customer: Optional[int] = None      # 前一GB最后客户（全局id）
    next_gb_center: Optional[Tuple[float, float]] = None  # 下一GB中心坐标(x,y)


@dataclass
class ACOResult:
    order: List[int]       # 全局节点序列（包含插入的 prev/depot）
    cost: float
    history: List[float]


# =========================
# Core: Ant
# =========================

class Ant:
    def __init__(self, local_nodes: List[int], tau: np.ndarray, eta: np.ndarray, rng: np.random.Generator):
        self.local_nodes = list(local_nodes)
        self.tau = tau
        self.eta = eta
        self.rng = rng

        start = int(rng.choice(self.local_nodes))
        self.route = [start]

        self.visited = {start}
        self.unvisited = list(set(self.local_nodes) - self.visited)
        self.cur = start

        self.cost = float("inf")

    def select_next(self, alpha: float, beta: float, greedy_prob: float) -> Optional[int]:
        if not self.unvisited:
            return None

        cand = np.array(self.unvisited, dtype=int)
        tau = self.tau[self.cur, cand]
        eta = self.eta[self.cur, cand]

        score = np.power(tau, alpha) * np.power(eta, beta)
        s = float(score.sum())

        if not np.isfinite(s) or s <= 0:
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
# Core: Customer ACO Solver
# =========================

class CustomerACO:
    """
    只负责：给定一个 customer_id 列表（全局id），输出其访问顺序（可插 prev/depot/next-center 影响cost）
    """

    def __init__(self, instance, customers: List[int], context: RouteContext, params: ACOParams):
        if not customers:
            raise ValueError("customers is empty")

        self.instance = instance
        self.customers_global = list(map(int, customers))
        self.context = context
        self.params = params

        self.rng = np.random.default_rng(params.seed)

        # ---- 全局 -> 局部映射（局部编号 0..n-1） ----
        self.n = len(self.customers_global)
        self.g2l = {cid: i for i, cid in enumerate(self.customers_global)}
        self.l2g = {i: cid for cid, i in self.g2l.items()}

        # ---- 局部距离矩阵 ----
        dm = self.instance.distance_matrix
        idx = np.array(self.customers_global, dtype=int)
        self.D = dm[np.ix_(idx, idx)].astype(float)

        self.tau = np.ones((self.n, self.n), dtype=float)
        self.eta = 1.0 / (self.D + 1e-6)
        np.fill_diagonal(self.tau, 0.0)
        np.fill_diagonal(self.eta, 0.0)

        self.best_route_local: Optional[List[int]] = None
        self.best_cost: float = float("inf")

    def _nearest_depot(self, node_global: int) -> int:
        depots = np.array(self.instance.depots, dtype=int)
        dists = self.instance.distance_matrix[depots, node_global]
        return int(depots[int(np.argmin(dists))])

    def _apply_context_insertions(self, route_global: List[int]) -> List[int]:
        r = list(route_global)

        # 插入 prev_last_customer
        if self.context.prev_last_customer is not None:
            r.insert(0, int(self.context.prev_last_customer))

        # 插 depot
        mode = self.context.depot_mode
        if mode is not None:
            if mode in ("depart", "depart & return"):
                dep = self._nearest_depot(r[0])
                r.insert(0, dep)
            if mode in ("return", "depart & return"):
                dep = self._nearest_depot(r[-1])
                r.append(dep)

        return r

    def evaluate(self, route_local: List[int]) -> float:
        # local -> global（只含客户）
        route_global = [self.l2g[i] for i in route_local]

        # 插入 prev/depot
        full = self._apply_context_insertions(route_global)

        # 累积距离（使用全局距离矩阵）
        dm = self.instance.distance_matrix
        total = 0.0
        for i in range(len(full) - 1):
            total += dm[full[i], full[i + 1]]

        # 可选：连接到 next GB center
        if self.context.next_gb_center is not None:
            last = self.instance.location(full[-1])
            total += float(np.linalg.norm(np.array(last, float) - np.array(self.context.next_gb_center, float)))

        return float(total)

    def update_pheromone(self, ants: List[Ant], elite_route: List[int], elite_cost: float) -> None:
        rho, eps = self.params.rho, self.params.epsilon
        delta = np.zeros_like(self.tau)

        for ant in ants:
            c = ant.cost
            if not np.isfinite(c) or c <= 0:
                continue
            r = ant.route
            for i in range(len(r) - 1):
                delta[r[i], r[i + 1]] += 1.0 / c

        elite = np.zeros_like(self.tau)
        elite_cost = elite_cost if elite_cost > 0 else 1e-6
        for i in range(len(elite_route) - 1):
            elite[elite_route[i], elite_route[i + 1]] += 1.0 / elite_cost

        self.tau = (1 - rho) * self.tau + delta + eps * elite
        np.fill_diagonal(self.tau, 0.0)

    def solve(self) -> ACOResult:
        history: List[float] = []
        local_nodes = list(range(self.n))

        for _ in range(self.params.num_iter):
            ants: List[Ant] = []

            for _k in range(self.params.num_ants):
                ant = Ant(local_nodes, self.tau, self.eta, self.rng)
                ant.construct(self.params.alpha, self.params.beta, self.params.greedy_prob)
                ant.cost = self.evaluate(ant.route)
                ants.append(ant)

                if ant.cost < self.best_cost:
                    self.best_cost = ant.cost
                    self.best_route_local = list(ant.route)

            if self.best_route_local is None:
                continue

            self.update_pheromone(ants, self.best_route_local, self.best_cost)
            history.append(self.best_cost)

        # 输出全局序列（含插入）
        best_local = self.best_route_local or list(range(self.n))
        best_global_customers = [self.l2g[i] for i in best_local]
        best_full = self._apply_context_insertions(best_global_customers)

        return ACOResult(order=best_global_customers, cost=self.best_cost, history=history)


# =========================
# Public API functions
# =========================

def plan_customers_order(instance, customers: List[int], *, context: Optional[RouteContext] = None,
                         params: Optional[ACOParams] = None) -> ACOResult:
    context = context or RouteContext()
    params = params or ACOParams()
    return CustomerACO(instance, customers, context, params).solve()



def plan_internal_gbs_order(
    instance,
    gb_order: List[int],                 # 外层GB访问顺序
    gb_centers: List[Tuple[float, float]],
    gb_customers: List[List[int]],       # 每个GB内部客户id
    *,
    params: ACOParams,
) -> List[List[int]]:
    """
    给定：
      - GB顺序
      - GB中心
      - GB内部客户集合

    返回：
      - 每个GB内部的最优客户路径（含 depot/串接）
    """

    routes = []

    for pos, gb_id in enumerate(gb_order):

        customers = gb_customers[gb_id]
        if not customers:
            routes.append([])
            continue

        # ---------- 构造上下文 ----------
        if len(gb_order) == 1:
            ctx = RouteContext(depot_mode="depart & return")

        elif pos == 0:
            ctx = RouteContext(
                depot_mode="depart",
                next_gb_center=gb_centers[gb_order[pos + 1]]
            )

        elif pos == len(gb_order) - 1:
            ctx = RouteContext(
                depot_mode="return",
                prev_last_customer=routes[-1][-1]
            )

        else:
            ctx = RouteContext(
                prev_last_customer=routes[-1][-1],
                next_gb_center=gb_centers[gb_order[pos + 1]]
            )

        # ---------- 规划当前GB内部路径 ----------
        res = plan_customers_order(
            instance,
            customers,
            context=ctx,
            params=params
        )

        routes.append(res.order)

    return routes


