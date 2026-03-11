from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal
import numpy as np

from params import GAParams



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
    order: List[int]       # 当前GB内部 customers 的全局id顺序（不含 prev/depot）
    cost: float
    history: List[float]


# =========================
# Core: Ant
# =========================
class CustomerGA:
    """
    给定 customers(全局id)，输出其访问顺序（考虑 prev/depot/next-center 对 cost 的影响）
    """

    def __init__(self, instance, customers: List[int], context: RouteContext, params: GAParams):
        if not customers:
            raise ValueError("customers is empty")

        self.instance = instance
        self.customers_global = list(map(int, customers))
        self.context = context
        self.params = params
        self.rng = np.random.default_rng(params.seed)

        # ---- 全局 -> 局部映射 ----
        self.n = len(self.customers_global)
        self.g2l = {cid: i for i, cid in enumerate(self.customers_global)}
        self.l2g = {i: cid for cid, i in self.g2l.items()}

        # ---- 局部距离矩阵（如果你之后想加 2-opt/启发式会用到）----
        dm = self.instance.distance_matrix
        idx = np.array(self.customers_global, dtype=int)
        self.D = dm[np.ix_(idx, idx)].astype(float)

        self.best_route_local: Optional[np.ndarray] = None
        self.best_cost: float = float("inf")

    def _nearest_depot(self, node_global: int) -> int:
        depots = np.array(self.instance.depots, dtype=int)
        dists = self.instance.distance_matrix[depots, node_global]
        return int(depots[int(np.argmin(dists))])

    def _apply_context_insertions(self, route_global: List[int]) -> List[int]:
        r = list(route_global)

        if self.context.prev_last_customer is not None:
            r.insert(0, int(self.context.prev_last_customer))

        mode = self.context.depot_mode
        if mode is not None:
            if mode in ("depart", "depart & return"):
                dep = self._nearest_depot(r[0])
                r.insert(0, dep)
            if mode in ("return", "depart & return"):
                dep = self._nearest_depot(r[-1])
                r.append(dep)

        return r

    def evaluate(self, route_local: np.ndarray) -> float:
        # local -> global（只含客户）
        route_global = [self.l2g[int(i)] for i in route_local]

        # 插入 prev/depot
        full = self._apply_context_insertions(route_global)

        # 累积距离（全局距离矩阵）
        dm = self.instance.distance_matrix
        total = 0.0
        for i in range(len(full) - 1):
            total += dm[full[i], full[i + 1]]

        # 连接到 next GB center
        if self.context.next_gb_center is not None:
            last = self.instance.location(full[-1])
            total += float(
                np.linalg.norm(
                    np.array(last, float) - np.array(self.context.next_gb_center, float)
                )
            )

        return float(total)

    # ---------------- GA operators ----------------
    def init_population(self) -> np.ndarray:
        pop = np.empty((self.params.pop_size, self.n), dtype=int)
        base = np.arange(self.n, dtype=int)
        for i in range(self.params.pop_size):
            pop[i] = self.rng.permutation(base)
        return pop

    def tournament_select(self, pop: np.ndarray, costs: np.ndarray) -> np.ndarray:
        k = min(int(self.params.tournament_k), pop.shape[0])
        idx = self.rng.integers(0, pop.shape[0], size=k)
        best = idx[int(np.argmin(costs[idx]))]
        return pop[best].copy()

    def ox_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        n = p1.size
        if n <= 1:
            return p1.copy()

        a, b = sorted(self.rng.choice(n, size=2, replace=False))
        child = np.full(n, -1, dtype=int)
        child[a:b+1] = p1[a:b+1]
        fill = [x for x in p2 if x not in child]
        ptr = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill[ptr]
                ptr += 1
        return child

    def mutate(self, x: np.ndarray) -> None:
        n = x.size
        if n <= 2:
            return
        if self.rng.random() < self.params.inversion_rate:
            i, j = sorted(self.rng.choice(n, size=2, replace=False))
            x[i:j+1] = x[i:j+1][::-1]
        else:
            i, j = self.rng.choice(n, size=2, replace=False)
            x[i], x[j] = x[j], x[i]

    # ---------------- solve ----------------
    def solve(self) -> ACOResult:
        history: List[float] = []

        # ---- 边界：0/1 个客户 ----
        if self.n <= 1:
            if self.n == 0:
                return ACOResult(order=[], cost=0.0, history=[0.0])
            # n==1
            route = np.array([0], dtype=int)
            c = self.evaluate(route)
            return ACOResult(order=[self.l2g[0]], cost=float(c), history=[float(c)])
    
        pop = self.init_population()
        costs = np.array([self.evaluate(ind) for ind in pop], dtype=float)

        best_idx = int(np.argmin(costs))
        self.best_route_local = pop[best_idx].copy()
        self.best_cost = float(costs[best_idx])
        history.append(self.best_cost)

        for _gen in range(self.params.num_gen):
            elite_size = max(0, int(self.params.elite_size))
            elite_size = min(elite_size, self.params.pop_size)
            elite_idx = np.argsort(costs)[:elite_size]
            elites = pop[elite_idx].copy()

            new_pop = []
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

            gen_best = float(costs.min())
            if gen_best < self.best_cost:
                idx = int(np.argmin(costs))
                self.best_cost = gen_best
                self.best_route_local = pop[idx].copy()

            history.append(self.best_cost)

        best_local = self.best_route_local if self.best_route_local is not None else np.arange(self.n)
        best_global_customers = [self.l2g[int(i)] for i in best_local]

        # ⚠️ 这里我保持和你现在 plan_internal_gbs_order 的用法一致：返回“仅客户顺序”
        return ACOResult(order=best_global_customers, cost=self.best_cost, history=history)



# =========================
# Public API functions
# =========================

def plan_customers_order(instance, customers: List[int], *, context: Optional[RouteContext] = None,
                         params: Optional[GAParams] = None) -> ACOResult:
    context = context or RouteContext()
    params = params or GAParams()
    return CustomerGA(instance, customers, context, params).solve()



def plan_internal_gbs_order_ga(
    instance,
    gb_order: List[int],                 # 外层GB访问顺序
    gb_centers: List[Tuple[float, float]],
    gb_customers: List[List[int]],       # 每个GB内部客户id
    *,
    params: GAParams,
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


