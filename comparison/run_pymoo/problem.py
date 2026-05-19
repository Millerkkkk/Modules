import os
import sys
import time
from matplotlib.pylab import norm
import numpy as np

from EVRP.loader import load_problem
from evaluate import Evaluator
from params import ev_params
from stochastic_demand import fast_forward_selection, generate_normal_scenarios

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pymoo.core.sampling import Sampling
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination

from pymoo.util.display.output import Output
from pymoo.util.display.column import Column
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.visualization.scatter import Scatter




def max_expected_load(vehicle_capacity, beta, alpha):
    z = norm.ppf(alpha)
    return vehicle_capacity / (1 + z * beta)



# termination = get_termination("n_gen", 2000)
termination = DefaultMultiObjectiveTermination(
    period=50,       # 50代无改进
    n_max_gen=2000   # 最大代数
)

# termination = ("time", 240)


# comparison/run_pymoo/gb_cmopso_utils.py

import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from typing import Dict, List, Tuple

# -------------------
# 结合GB框架
# --------------------


@dataclass
class GBBlockState:
    template_routes_clusters: Dict[int, List[List[int]]]

    # block 列表，每个 block 是一个 GB
    # [(cid, gb_idx, [cust...]), ...]
    blocks: List[Tuple[int, int, List[int]]]

    # block 索引映射
    block_to_var_idx: Dict[Tuple[int, int], int]

    # 所有客户
    all_customers: List[int]
    cust_to_var_idx: Dict[int, int]

    n_blocks: int
    n_customers: int
    n_var: int



def encode_routes_clusters(
    routes_clusters: Dict[int, List[List[int]]],
    state: GBBlockState
) -> np.ndarray:
    """
    把现有 routes_clusters 编成一个连续向量 x
    x 的长度 = 客户总数
    x 越小，表示该客户在本 GB 中越靠前
    """
    n_var = len(state.all_customers)
    x = np.zeros(n_var, dtype=float)

    # 给每个 GB 内的客户赋一个 rank
    for cid in sorted(routes_clusters.keys()):
        gb_list = routes_clusters[cid]
        for gb in gb_list:
            m = len(gb)
            if m == 1:
                x[state.cust_to_var_idx[gb[0]]] = 0.5
            else:
                for rank, cust in enumerate(gb):
                    x[state.cust_to_var_idx[cust]] = rank / (m - 1)

    return x


def decode_routes_clusters(
    x: np.ndarray,
    state: GBBlockState
) -> Dict[int, List[List[int]]]:
    """
    只重排每个 GB 内部顺序
    cluster 不变
    GB 块数量不变
    每个 GB 的客户集合不变
    """
    out = {cid: [] for cid in sorted(state.template_routes_clusters.keys())}

    for cid, gb_idx, gb in state.blocks:
        pairs = []
        for cust in gb:
            vidx = state.cust_to_var_idx[cust]
            pairs.append((float(x[vidx]), cust))

        pairs.sort(key=lambda t: t[0])
        new_gb = [cust for _, cust in pairs]
        out[cid].append(new_gb)

    return out


class GBClusterRoutingProblem(Problem):
    def __init__(self, state, evaluator_fn):
        n_var = len(state.all_customers)

        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_ieq_constr=0,
            xl=np.zeros(n_var, dtype=float),
            xu=np.ones(n_var, dtype=float),
        )

        self.state = state
        self.evaluator_fn = evaluator_fn

    def _evaluate(self, X, out, *args, **kwargs):
        n = X.shape[0]
        F = np.zeros((n, 2), dtype=float)

        for i in range(n):
            routes_clusters = decode_routes_clusters(X[i], self.state)
            cend = self.evaluator_fn(routes_clusters)

            F[i, 0] = float(cend["total_cost"])
            F[i, 1] = float(cend.get("total_ra", 0.0))

        out["F"] = F


class GBHeuristicSampling(Sampling):
    def __init__(self, state, seed_solutions=None):
        super().__init__()
        self.state = state
        self.seed_solutions = seed_solutions or []

    def _do(self, problem, n_samples, **kwargs):
        X = np.random.rand(n_samples, problem.n_var)

        n_seed = min(len(self.seed_solutions), n_samples)
        for i in range(n_seed):
            X[i, :] = encode_routes_clusters(self.seed_solutions[i], self.state)

        return X
    






# -----------
#把原来的 encode 换成双层 encode。
# -----------

def encode_routes_clusters_dual(
    routes_clusters: Dict[int, List[List[int]]],
    state: GBBlockState
) -> np.ndarray:
    x = np.zeros(state.n_var, dtype=float)

    # --------
    # 1) block-level keys
    # --------
    for cid in sorted(routes_clusters.keys()):
        gb_list = routes_clusters[cid]
        m = len(gb_list)

        for order_idx, gb in enumerate(gb_list):
            # 需要找到这个 gb 在模板里的身份
            gb_set = tuple(sorted(gb))

            matched_gb_idx = None
            for tpl_cid, tpl_gb_idx, tpl_gb in state.blocks:
                if tpl_cid != cid:
                    continue
                if tuple(sorted(tpl_gb)) == gb_set:
                    matched_gb_idx = tpl_gb_idx
                    break

            if matched_gb_idx is None:
                raise ValueError(f"Cannot match GB block in cluster {cid}: {gb}")

            vidx = state.block_to_var_idx[(cid, matched_gb_idx)]
            x[vidx] = order_idx / max(m - 1, 1)

    # --------
    # 2) customer-level keys
    # --------
    offset = state.n_blocks
    for cid in sorted(routes_clusters.keys()):
        gb_list = routes_clusters[cid]
        for gb in gb_list:
            m = len(gb)
            if m == 1:
                x[offset + state.cust_to_var_idx[gb[0]]] = 0.5
            else:
                for order_idx, cust in enumerate(gb):
                    x[offset + state.cust_to_var_idx[cust]] = order_idx / (m - 1)

    return np.clip(x, 0.0, 1.0)


def decode_routes_clusters_dual(
    x: np.ndarray,
    state: GBBlockState
) -> Dict[int, List[List[int]]]:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, 1.0)

    out = {}

    block_part = x[:state.n_blocks]
    cust_part = x[state.n_blocks:]

    # 先按 cluster 收集 blocks
    cluster_blocks = {}
    for cid, gb_idx, gb in state.blocks:
        bvidx = state.block_to_var_idx[(cid, gb_idx)]
        bkey = float(block_part[bvidx])

        # GB 内部排序
        pairs = []
        for cust in gb:
            cvidx = state.cust_to_var_idx[cust]
            ckey = float(cust_part[cvidx])
            pairs.append((ckey, cust))

        pairs.sort(key=lambda t: t[0])
        sorted_gb = [cust for _, cust in pairs]

        cluster_blocks.setdefault(cid, []).append((bkey, gb_idx, sorted_gb))

    # 每个 cluster 内按 block key 排序
    for cid in sorted(cluster_blocks.keys()):
        gb_items = cluster_blocks[cid]
        gb_items.sort(key=lambda t: t[0])
        out[cid] = [gb for _, _, gb in gb_items]

    return out


class GBClusterRoutingProblemDual(Problem):
    def __init__(self, state, evaluator_fn):
        super().__init__(
            n_var=state.n_var,
            n_obj=2,
            n_ieq_constr=0,
            xl=np.zeros(state.n_var, dtype=float),
            xu=np.ones(state.n_var, dtype=float),
        )
        self.state = state
        self.evaluator_fn = evaluator_fn

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X, dtype=float)
        X = np.clip(X, 0.0, 1.0)

        F = np.zeros((X.shape[0], 2), dtype=float)

        for i in range(X.shape[0]):
            routes_clusters = decode_routes_clusters_dual(X[i], self.state)
            cend = self.evaluator_fn(routes_clusters)

            F[i, 0] = float(cend["total_cost"])
            F[i, 1] = float(cend.get("total_ra", 0.0))

        out["F"] = F


class GBHeuristicSamplingDual(Sampling):
    def __init__(self, state, seed_solutions=None):
        super().__init__()
        self.state = state
        self.seed_solutions = seed_solutions or []

    def _do(self, problem, n_samples, **kwargs):
        X = np.random.rand(n_samples, problem.n_var)

        n_seed = min(len(self.seed_solutions), n_samples)
        for i in range(n_seed):
            X[i, :] = encode_routes_clusters_dual(
                self.seed_solutions[i],
                self.state
            )

        X = np.clip(X, 0.0, 1.0)
        if not np.all(np.isfinite(X)):
            raise ValueError("Sampling produced NaN/Inf.")

        return X






class MDEVRPProblem(Problem):

    def __init__(self, problem, evaluator, vehicle_capacity, scenarios):
        self.problem = problem
        self.evaluator = evaluator
        if vehicle_capacity == None:
            self.vehicle_capacity = self.problem.vehicle_capacity
        else:
            self.vehicle_capacity = vehicle_capacity
        self.scenarios = scenarios
        self.customer_ids = list(problem.customers)

        n = len(self.customer_ids)

        super().__init__(
            n_var=n,
            n_obj=2,
            n_ieq_constr=0,
            xl=0,
            xu=n - 1,
            vtype=int
        )

    def decode(self, x):
        x = np.asarray(x, dtype=int)
        return [self.customer_ids[i] for i in x]

    def split_routes(self, sequence):
        routes = []
        current_route = []
        current_load = 0.0
        cap = self.vehicle_capacity

        for customer in sequence:
            demand = self.problem.demand[customer]

            if current_load + demand <= cap:
                current_route.append(customer)
                current_load += demand
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [customer]
                current_load = demand

        if current_route:
            routes.append(current_route)

        return {i: r for i, r in enumerate(routes)}

    def _evaluate(self, X, out, *args, **kwargs):
        F = []

        for x in X:
            sequence = self.decode(x)
            routes_by_clusters = self.split_routes(sequence)
            # result = self.evaluator.evaluate_saa(routes_by_clusters, self.scenarios)

            result = self.evaluator.evaluate_clusters(routes_by_clusters)

            total_cost = float(result["total_cost"])
            total_ra = float(result["total_ra"])
            F.append([total_cost, total_ra])

        out["F"] = np.array(F)

    def evaluate_solution(self, x):
        sequence = self.decode(x)
        routes_by_clusters = self.split_routes(sequence)
        # result = self.evaluator.evaluate_saa(routes_by_clusters, self.scenarios)
        result = self.evaluator.evaluate_clusters(routes_by_clusters)
        return sequence, routes_by_clusters, result


class MyOutput(MultiObjectiveOutput):

    def __init__(self):
        super().__init__()
        

        self.best_cost = Column("best_cost", width=13)
        self.best_ra = Column("best_ra", width=13)

        cols = list(self.columns)
        
        indicator_index = next((i for i, c in enumerate(cols) if c.name == "indicator"), len(cols))
        

        self.columns = (
            cols[:indicator_index + 1]
            + [self.best_cost, self.best_ra]
            + cols[indicator_index + 1:]
        )

    def update(self, algorithm):
        super().update(algorithm)

        F = algorithm.pop.get("F")
        self.best_cost.set(np.min(F[:, 0]))
        self.best_ra.set(np.min(F[:, 1]))



def load_instance(file_path, expected_vehicle_capacity, 
                  beta=0.1, n_scenarios=10, seed=42):
    # file_path = os.path.join(base_path, f"{instance_name}.txt")
    problem = load_problem(file_path, ev_params)

    mu = np.asarray(problem.demand, dtype=float)
    scenarios = generate_normal_scenarios(
        mu=mu,
        beta=beta,
        n_scenarios=n_scenarios,
        random_state=seed,
    )

    evaluator = Evaluator(problem, ra_safe=0.1, ra_risk=0.7)

    md_evrp_problem = MDEVRPProblem(
        problem=problem,
        evaluator=evaluator,
        vehicle_capacity=expected_vehicle_capacity,
        scenarios=scenarios
    )

    return md_evrp_problem


def load_instance_ffs(file_path, 
                  beta=0.1, n_scenarios=100, n_reduced_scenarios=5, seed=42):
    # file_path = os.path.join(base_path, f"{instance_name}.txt")
    problem = load_problem(file_path, ev_params)

    mu = np.asarray(problem.demand, dtype=float)
    scenarios = generate_normal_scenarios(
        mu=mu,
        beta=beta,
        n_scenarios=n_scenarios,
        random_state=seed,
    )

    selected_idx, reduced_scenarios, reduced_probs = fast_forward_selection(
        scenarios=scenarios,
        n_select=n_reduced_scenarios,
    )

    evaluator = Evaluator(problem, ra_safe=0.1, ra_risk=0.7)

    md_evrp_problem = MDEVRPProblem_ffs(
        problem=problem,
        evaluator=evaluator,
        vehicle_capacity=None,
        scenarios=reduced_scenarios
    )

    return md_evrp_problem


class MDEVRPProblem_ffs(Problem):

    def __init__(self, problem, evaluator, vehicle_capacity, scenarios):
        self.problem = problem
        self.evaluator = evaluator
        if vehicle_capacity == None:
            self.vehicle_capacity = self.problem.vehicle_capacity
        else:
            self.vehicle_capacity = vehicle_capacity
        self.scenarios = scenarios
        self.customer_ids = list(problem.customers)

        n = len(self.customer_ids)

        super().__init__(
            n_var=n,
            n_obj=2,
            n_ieq_constr=0,
            xl=0,
            xu=n - 1,
            vtype=int
        )

    def decode(self, x):
        x = np.asarray(x, dtype=int)
        return [self.customer_ids[i] for i in x]

    def split_routes(self, sequence):
        routes = []
        current_route = []
        current_load = 0.0
        cap = self.vehicle_capacity

        for customer in sequence:
            demand = self.problem.demand[customer]

            if current_load + demand <= cap:
                current_route.append(customer)
                current_load += demand
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [customer]
                current_load = demand

        if current_route:
            routes.append(current_route)

        return {i: r for i, r in enumerate(routes)}

    def _evaluate(self, X, out, *args, **kwargs):
        F = []

        for x in X:
            sequence = self.decode(x)
            routes_by_clusters = self.split_routes(sequence)
            # result = self.evaluator.evaluate_saa(routes_by_clusters, self.scenarios)

            result = self.evaluator.evaluate_clusters(routes_by_clusters)

            total_cost = float(result["total_cost"])
            total_ra = float(result["total_ra"])
            F.append([total_cost, total_ra])

        out["F"] = np.array(F)

    def evaluate_solution(self, x):
        sequence = self.decode(x)
        routes_by_clusters = self.split_routes(sequence)
        # result = self.evaluator.evaluate_saa(routes_by_clusters, self.scenarios)
        result = self.evaluator.evaluate_clusters(routes_by_clusters)
        return sequence, routes_by_clusters, result



class BestObjectiveCallback(Callback):

    def __init__(self):
        super().__init__()

    def notify(self, algorithm):

        F = algorithm.pop.get("F")

        best_cost = np.min(F[:, 0])
        best_ra = np.min(F[:, 1])

        gen = algorithm.n_gen

        print(
            f"Gen {gen:4d} | Best Cost: {best_cost:.4f} | Best RA: {best_ra:.4f}"
        )


def solve(md_evrp_problem, algorithm, seed=1, verbose=True):
    callback = BestObjectiveCallback()

    return minimize(
        md_evrp_problem,
        algorithm,
        termination=termination,
        seed=seed,
        verbose=verbose,
        output=MyOutput()
    )





def plot_pareto(res, instance_name):
    if res.F is None:
        print("No Pareto front to plot.")
        return

    plot = Scatter(title=f"Pareto Front - {instance_name}")
    plot.add(res.F, label="NSGA3")
    plot.show()

    
def print_summary(res, instance_name, run_time):
    print("=" * 60)
    print(f"Instance: {instance_name}")
    print("Pareto objective values:")
    print(res.F)
    print("Run time:", run_time)


def print_decoded_solutions(md_evrp_problem, res, max_solutions=None):
    print("=" * 60)
    if res.X is None:
        print("No solutions found.")
        return

    X = res.X
    if max_solutions is not None:
        X = X[:max_solutions]

    for i, x in enumerate(X):
        sequence, routes_by_clusters, result = md_evrp_problem.evaluate_solution(x)

        print(f"\nSolution #{i}")
        print("Sequence:", sequence)
        print("Routes by clusters:", routes_by_clusters)
        print("Total cost:", result["total_cost"])
        print("Total RA:", result["total_ra"])
        print("Stage1 cost:", result.get("stage1_cost"))
        print("Stage2 cost:", result.get("stage2_cost"))



