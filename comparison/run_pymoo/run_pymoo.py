from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import copy
import random
import numpy as np

from Clustering.api import clustering
from EVRP.loader import load_problem
from CustomerPlanning_GA import plan_internal_gbs_order_ga
from GB.api import gb_clustering
from GbPlanning_GA import plan_gb_order_ga
from params import GAParams

from evaluate import Evaluator
from stochastic_demand import fast_forward_selection, generate_normal_scenarios

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination


# ============================================================
# 1. Config
# ============================================================

@dataclass
class ExperimentConfig:
    instance_path: str
    algorithm: str = "nsga2"
    base_seed: int = 42
    run_id: int = 0
    pop_size: int = 100
    n_eval: int = 5000
    n_solutions: int = 20
    n_scenarios: int = 100
    n_reduced_scenarios: int = 10
    beta: float = 0.20
    alpha: float = 0.50
    verbose: bool = True

    ra_safe: float = 0.10
    ra_risk: float = 0.70

    time_limit: str = "00:20:00"   # 1200 秒


# ============================================================
# 2. Domain hooks
# ============================================================

def generate_init_solution(problem, base_seed: int):
    depot_coords = np.asarray(
        [(problem.nodes[i].x, problem.nodes[i].y) for i in problem.depots],
        dtype=float,
    )

    clusters = clustering(
        problem,
        method="kmeans",
        feature="location",
        random_state=base_seed,
    )

    routes_clusters = {}

    for cid, cluster in clusters.items():
        customer_ids = list(cluster)
        points = np.asarray(
            [(problem.nodes[i].x, problem.nodes[i].y) for i in customer_ids],
            dtype=float,
        )

        gbs_info = gb_clustering(
            points,
            method="gb_kmeans",
            split_k=0.6,
            merge=True,
            random_state=base_seed + cid,
        )

        gbs_customer_id = [[customer_ids[j] for j in cl] for cl in gbs_info.gbs]

        params_gb = GAParams(
            pop_size=30,
            num_gen=100,
            tournament_k=3,
            crossover_rate=0.9,
            mutation_rate=0.2,
            inversion_rate=0.7,
            elite_size=2,
            seed=base_seed + 1000 + cid,
        )
        gb_order_result = plan_gb_order_ga(
            gbs_info.centers,
            depot_coords,
            params_gb,
        )

        params_customer = GAParams(
            pop_size=60,
            num_gen=100,
            tournament_k=3,
            crossover_rate=0.9,
            mutation_rate=0.3,
            inversion_rate=0.7,
            elite_size=2,
            seed=base_seed + 2000 + cid,
        )
        routes = plan_internal_gbs_order_ga(
            problem,
            gb_order_result.order,
            gbs_info.centers,
            gbs_customer_id,
            params=params_customer,
        )

        routes_clusters[cid] = routes

    return routes_clusters


def canonicalize_routes_clusters(routes_clusters):
    return tuple(
        (cid, tuple(tuple(route) for route in routes))
        for cid, routes in sorted(routes_clusters.items())
    )


def perturb_solution(routes_clusters, rng=None):
    if rng is None:
        rng = random.Random()

    sol = copy.deepcopy(routes_clusters)
    cluster_ids = sorted(sol.keys())

    cid = rng.choice(cluster_ids)
    routes = sol[cid]

    if len(routes) == 0:
        return sol

    move_type = rng.choice([
        "swap_blocks",
        "reverse_block_segment",
        "insert_block",
        "reverse_inside_block",
        "swap_inside_block",
    ])

    if move_type == "swap_blocks" and len(routes) >= 2:
        i, j = sorted(rng.sample(range(len(routes)), 2))
        routes[i], routes[j] = routes[j], routes[i]

    elif move_type == "reverse_block_segment" and len(routes) >= 2:
        i, j = sorted(rng.sample(range(len(routes)), 2))
        routes[i:j+1] = list(reversed(routes[i:j+1]))

    elif move_type == "insert_block" and len(routes) >= 2:
        i, j = rng.sample(range(len(routes)), 2)
        block = routes.pop(i)
        routes.insert(j, block)

    elif move_type == "reverse_inside_block":
        candidate_idx = [k for k, block in enumerate(routes) if len(block) >= 2]
        if candidate_idx:
            k = rng.choice(candidate_idx)
            routes[k] = list(reversed(routes[k]))

    elif move_type == "swap_inside_block":
        candidate_idx = [k for k, block in enumerate(routes) if len(block) >= 2]
        if candidate_idx:
            k = rng.choice(candidate_idx)
            block = routes[k]
            i, j = rng.sample(range(len(block)), 2)
            block[i], block[j] = block[j], block[i]

    sol[cid] = routes
    return sol


def flatten_routes_clusters(routes_clusters):
    flat = {}
    for cid, blocks in routes_clusters.items():
        seq = []
        for block in blocks:
            seq.extend(block)
        flat[cid] = seq
    return flat


def generate_init_population(core_solution, pop_size, seed=42, max_trials=5000):
    rng = random.Random(seed)

    population = [copy.deepcopy(core_solution)]
    seen = {canonicalize_routes_clusters(core_solution)}

    trials = 0
    while len(population) < pop_size and trials < max_trials:
        new_sol = perturb_solution(core_solution, rng=rng)
        key = canonicalize_routes_clusters(new_sol)

        if key not in seen:
            seen.add(key)
            population.append(new_sol)

        trials += 1

    if len(population) < pop_size:
        print(f"[WARN] only generated {len(population)} unique solutions, target={pop_size}")

    population_flat = [flatten_routes_clusters(sol) for sol in population]
    return population, population_flat


# ============================================================
# 3. Flat encoding helpers
# ============================================================

def build_flat_encoding_state(seed_solution):
    cluster_ids = sorted(seed_solution.keys())

    cluster_lengths = {}
    cluster_nodes = {}
    slices = {}

    start = 0
    for cid in cluster_ids:
        nodes = list(seed_solution[cid])
        length = len(nodes)

        cluster_lengths[cid] = length
        cluster_nodes[cid] = set(nodes)
        slices[cid] = (start, start + length)
        start += length

    return {
        "cluster_ids": cluster_ids,
        "cluster_lengths": cluster_lengths,
        "cluster_nodes": cluster_nodes,
        "slices": slices,
        "n_var": start,
    }


def encode_flat_routes(solution, state):
    x = []
    for cid in state["cluster_ids"]:
        x.extend(list(solution[cid]))
    return np.asarray(x, dtype=int)


def decode_flat_routes(x, state):
    x = np.asarray(x).astype(int)
    sol = {}
    for cid in state["cluster_ids"]:
        s, e = state["slices"][cid]
        sol[cid] = list(map(int, x[s:e]))
    return sol


def repair_flat_routes(sol, state):
    repaired = {}

    for cid in state["cluster_ids"]:
        target_nodes = state["cluster_nodes"][cid]
        seq = sol[cid]

        seen = set()
        filtered = []

        for node in seq:
            if node in target_nodes and node not in seen:
                filtered.append(node)
                seen.add(node)

        # 用固定顺序补全，避免 set 带来的不稳定顺序
        missing = [node for node in sorted(target_nodes) if node not in seen]
        filtered.extend(missing)

        repaired[cid] = filtered

    return repaired


def decode_and_repair(x, state):
    sol = decode_flat_routes(x, state)
    sol = repair_flat_routes(sol, state)
    return sol


# ============================================================
# 4. Problem / Sampling
# ============================================================

class GBClusterRoutingProblemFlat(Problem):
    def __init__(self, state, evaluator_fn):
        super().__init__(
            n_var=state["n_var"],
            n_obj=2,
            n_ieq_constr=0,
            xl=np.zeros(state["n_var"], dtype=int),
            xu=np.full(state["n_var"], 10**6, dtype=int),
            vtype=int,
        )
        self.state = state
        self.evaluator_fn = evaluator_fn

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[None, :]

        F = []
        for x in X:
            sol = decode_and_repair(x, self.state)
            f1, f2 = self.evaluator_fn(sol)
            F.append([float(f1), float(f2)])

        out["F"] = np.asarray(F, dtype=float)


class GBHeuristicSamplingFlat(Sampling):
    def __init__(self, state, seed_solutions, random_fill=True, seed: Optional[int] = None):
        super().__init__()
        self.state = state
        self.seed_solutions = list(seed_solutions)
        self.random_fill = random_fill
        self.rng = np.random.default_rng(seed)

    def _random_perturb(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()

        cid = int(self.rng.choice(self.state["cluster_ids"]))
        s, e = self.state["slices"][cid]
        seg_len = e - s

        if seg_len >= 2:
            i, j = self.rng.choice(np.arange(s, e), size=2, replace=False)
            y[i], y[j] = y[j], y[i]

        return y

    def _do(self, problem, n_samples, **kwargs):
        X = []

        for sol in self.seed_solutions[:n_samples]:
            X.append(encode_flat_routes(sol, self.state))

        if len(X) == 0:
            raise ValueError("seed_solutions is empty.")

        while len(X) < n_samples:
            base = X[len(X) % len(X)]
            if self.random_fill:
                X.append(self._random_perturb(base))
            else:
                X.append(base.copy())

        return np.asarray(X, dtype=int)


# ============================================================
# 5. Debug callback
# ============================================================

class DebugCallback(Callback):
    def __init__(self):
        super().__init__()
        self.data["history"] = []

    @staticmethod
    def _uniq_rows(a: Optional[np.ndarray], decimals: int = 8) -> int:
        if a is None:
            return 0
        a = np.asarray(a)
        if a.ndim == 1:
            a = a[None, :]
        return int(np.unique(np.round(a, decimals), axis=0).shape[0])

    def notify(self, algorithm):
        X = algorithm.pop.get("X")
        F = algorithm.pop.get("F")

        if F is not None:
            cost = F[:, 0]
            ra = F[:, 1]

            min_cost_idx = int(np.argmin(cost))
            min_ra_idx = int(np.argmin(ra))

            best_cost = float(cost[min_cost_idx])
            ra_of_best_cost = float(ra[min_cost_idx])

            best_ra = float(ra[min_ra_idx])
            cost_of_best_ra = float(cost[min_ra_idx])
        else:
            best_cost = np.nan
            ra_of_best_cost = np.nan
            best_ra = np.nan
            cost_of_best_ra = np.nan

        row = {
            "gen": int(getattr(algorithm, "n_gen", -1)),
            "n_eval": int(getattr(getattr(algorithm, "evaluator", None), "n_eval", -1)),
            "best_cost": best_cost,
            "ra_of_best_cost": ra_of_best_cost,
            "best_ra": best_ra,
            "cost_of_best_ra": cost_of_best_ra,
        }

        self.data["history"].append(row)


# ============================================================
# 6. Algorithm builders
# ============================================================

def build_algorithm_nsga2(pop_size: int, sampling: Sampling):
    from pymoo.operators.crossover.ox import OrderCrossover
    from pymoo.operators.mutation.inversion import InversionMutation

    return NSGA2(
        pop_size=pop_size,
        sampling=sampling,
        crossover=OrderCrossover(),
        mutation=InversionMutation(),
        eliminate_duplicates=True,
    )


def build_algorithm_moead(pop_size: int, sampling: Sampling):
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.operators.crossover.ox import OrderCrossover
    from pymoo.operators.mutation.inversion import InversionMutation

    ref_dirs = get_reference_directions("uniform", 2, n_partitions=max(4, pop_size // 10))

    return MOEAD(
        ref_dirs=ref_dirs,
        sampling=sampling,
        crossover=OrderCrossover(),
        mutation=InversionMutation(),
        n_neighbors=min(15, len(ref_dirs)),
        prob_neighbor_mating=0.9,
    )


def build_algorithm_nsga3(pop_size: int, sampling: Sampling):
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.util.ref_dirs import get_reference_directions

    ref_dirs = get_reference_directions("uniform", 2, n_partitions=max(4, pop_size // 10))

    return NSGA3(
        ref_dirs=ref_dirs,
        pop_size=len(ref_dirs),
        sampling=sampling,
    )


def build_algorithm_mopso(pop_size: int, sampling: Sampling, seed: Optional[int] = None):
    from pymoo.algorithms.moo.cmopso import CMOPSO

    return CMOPSO(
        pop_size=pop_size,
        max_velocity_rate=0.2,
        elite_size=max(10, pop_size // 5),
        mutation_rate=0.5,
        sampling=sampling,
        seed=seed,
    )


def build_algorithm(name: str, pop_size: int, sampling: Sampling, seed: Optional[int] = None):
    name = name.lower()

    if name == "nsga2":
        return build_algorithm_nsga2(pop_size=pop_size, sampling=sampling)

    if name == "nsga3":
        return build_algorithm_nsga3(pop_size=pop_size, sampling=sampling)

    if name == "moead":
        return build_algorithm_moead(pop_size=pop_size, sampling=sampling)

    if name in {"mopso", "cmopso"}:
        return build_algorithm_mopso(pop_size=pop_size, sampling=sampling, seed=seed)

    raise ValueError(f"Unknown algorithm: {name}")


# ============================================================
# 7. Shared experiment preparation
# ============================================================

def prepare_experiment_components(config: ExperimentConfig, ev_params: Any) -> Dict[str, Any]:
    seed = int(config.base_seed + config.run_id)

    raw_problem = load_problem(config.instance_path, ev_params)
    mu = np.asarray(raw_problem.demand, dtype=float)

    scenarios = generate_normal_scenarios(
        mu=mu,
        beta=config.beta,
        n_scenarios=config.n_scenarios,
        random_state=seed,
    )

    selected_idx, reduced_scenarios, reduced_probs = fast_forward_selection(
        scenarios=scenarios,
        n_select=config.n_reduced_scenarios,
    )

    evaluator_obj = Evaluator(raw_problem, ra_safe=config.ra_safe, ra_risk=config.ra_risk)

    def evaluator_fn(routes_clusters):
        vals = evaluator_obj.evaluate_ffs(routes_clusters, reduced_scenarios, reduced_probs)
        return float(vals["total_cost"]), float(vals["total_ra"])

    init_solution = generate_init_solution(
        problem=raw_problem,
        base_seed=seed,
    )

    init_vals = evaluator_obj.evaluate_ffs(init_solution, reduced_scenarios, reduced_probs)

    _, population_flat = generate_init_population(
        init_solution,
        pop_size=config.pop_size,
        seed=seed,
    )

    seed_solutions = population_flat
    state = build_flat_encoding_state(seed_solutions[0])

    sample_x = encode_flat_routes(seed_solutions[0], state)
    assert len(sample_x) == state["n_var"], "Encoded length does not match state['n_var']"

    mo_problem = GBClusterRoutingProblemFlat(
        state=state,
        evaluator_fn=evaluator_fn,
    )

    sampling = GBHeuristicSamplingFlat(
        state=state,
        seed_solutions=seed_solutions,
        random_fill=True,
        seed=seed,
    )

    return {
        "seed": seed,
        "raw_problem": raw_problem,
        "state": state,
        "sampling": sampling,
        "problem": mo_problem,
        "seed_solutions": seed_solutions,
        "reduced_scenarios": reduced_scenarios,
        "reduced_probs": reduced_probs,
        "selected_idx": selected_idx,
        "init_cost": float(init_vals["total_cost"]),
        "init_ra": float(init_vals["total_ra"]),
    }

# ============================================================
# 8. Result extraction
# ============================================================

def extract_result(
    config: ExperimentConfig,
    res,
    runtime_sec: float,
    debug_callback: DebugCallback,
    state: Any,
    init_cost: float,
    init_ra: float,
    selected_idx: Any,
) -> Dict[str, Any]:
    F = res.F
    X = res.X

    if X is None:
        pareto_routes = []
    elif np.ndim(X) == 1:
        pareto_routes = [decode_and_repair(X, state)]
    else:
        pareto_routes = [decode_and_repair(x, state) for x in X]

    if F is None:
        F_array = np.empty((0, 2))
    else:
        F_array = np.atleast_2d(F)

    pareto_points = []
    for row in F_array:
        if len(row) >= 2:
            pareto_points.append((float(row[0]), float(row[1])))

    if pareto_points:
        cost_values = np.array([p[0] for p in pareto_points], dtype=float)
        ra_values = np.array([p[1] for p in pareto_points], dtype=float)

        min_cost_idx = int(np.argmin(cost_values))
        min_ra_idx = int(np.argmin(ra_values))

        best_cost_in_archive = float(cost_values[min_cost_idx])
        ra_of_best_cost = float(ra_values[min_cost_idx])

        best_ra_in_archive = float(ra_values[min_ra_idx])
        cost_of_best_ra = float(cost_values[min_ra_idx])
    else:
        best_cost_in_archive = np.nan
        ra_of_best_cost = np.nan
        best_ra_in_archive = np.nan
        cost_of_best_ra = np.nan

    return {
        **asdict(config),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "instance": Path(config.instance_path).stem,
        "instance_path": str(Path(config.instance_path).resolve()),
        "seed": int(config.base_seed + config.run_id),

        "init_cost": float(init_cost),
        "init_ra": float(init_ra),

        "runtime_sec": float(runtime_sec),
        "n_gen": int(getattr(res.algorithm, "n_gen", -1)) if hasattr(res, "algorithm") else -1,
        "archive_size": int(len(F_array)),

        "best_cost_in_archive": best_cost_in_archive,
        "ra_of_best_cost": ra_of_best_cost,
        "best_ra_in_archive": best_ra_in_archive,
        "cost_of_best_ra": cost_of_best_ra,

        "pareto_points": pareto_points,
        "pareto_routes": pareto_routes,
        "X": X.tolist() if X is not None and hasattr(X, "tolist") else X,
        "F": F.tolist() if F is not None and hasattr(F, "tolist") else F,
        "debug_history": debug_callback.data["history"],
        "selected_idx": selected_idx.tolist() if hasattr(selected_idx, "tolist") else selected_idx,
    }

# ============================================================
# 9. Main runner
# ============================================================

def run_once(config: ExperimentConfig, ev_params: Any) -> Dict[str, Any]:
    components = prepare_experiment_components(config, ev_params)

    algorithm = build_algorithm(
        name=config.algorithm,
        pop_size=config.pop_size,
        sampling=components["sampling"],
        seed=components["seed"],
    )

    termination = get_termination("time", config.time_limit)
    debug_callback = DebugCallback()

    start = time.time()
    res = minimize(
        components["problem"],
        algorithm,
        termination=termination,
        seed=components["seed"],
        verbose=config.verbose,
        callback=debug_callback,
        save_history=False,
    )
    runtime_sec = time.time() - start

    return extract_result(
        config=config,
        res=res,
        runtime_sec=runtime_sec,
        debug_callback=debug_callback,
        state=components["state"],
        init_cost=components["init_cost"],
        init_ra=components["init_ra"],
        selected_idx=components["selected_idx"],
    )


import os
import json
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional


def _to_jsonable(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return str(obj)


def _save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, ensure_ascii=False, indent=2)


def _append_row_csv(path: str, record: Dict[str, Any]):
    df_row = pd.DataFrame([record])
    write_header = not os.path.exists(path)
    df_row.to_csv(path, mode="a", header=write_header, index=False, encoding="utf-8-sig")


def _list_instances(instance_source: str, recursive: bool = True, suffix: str = ".txt") -> List[str]:
    if not os.path.exists(instance_source):
        raise FileNotFoundError(f"instance_source not found: {instance_source}")

    if os.path.isfile(instance_source):
        if not instance_source.lower().endswith(suffix.lower()):
            raise ValueError(f"Single instance file must end with {suffix}: {instance_source}")
        return [os.path.abspath(instance_source)]

    files = []
    if recursive:
        for root, _, filenames in os.walk(instance_source):
            for f in filenames:
                if f.lower().endswith(suffix.lower()):
                    files.append(os.path.abspath(os.path.join(root, f)))
    else:
        for f in os.listdir(instance_source):
            fp = os.path.join(instance_source, f)
            if os.path.isfile(fp) and f.lower().endswith(suffix.lower()):
                files.append(os.path.abspath(fp))

    files = sorted(files)
    if len(files) == 0:
        raise FileNotFoundError(f"No {suffix} files found under: {instance_source}")

    return files


def _make_key(instance: str, algorithm: str, beta: float, run_id: int):
    return (str(instance), str(algorithm).lower(), round(float(beta), 6), int(run_id))



def _extract_progress_record_pymoo(rec: Dict[str, Any]) -> Dict[str, Any]:
    keep_cols = [
        "timestamp",
        "instance",
        "instance_path",
        "algorithm",
        "beta",
        "run_id",
        "seed",
        "time_limit",

        "pop_size",
        "n_eval",
        "n_solutions",
        "n_scenarios",
        "n_reduced_scenarios",

        "init_cost",
        "init_ra",

        "runtime_sec",
        "n_gen",
        "archive_size",

        "best_cost_in_archive",
        "ra_of_best_cost",
        "best_ra_in_archive",
        "cost_of_best_ra",
    ]

    out = {c: rec.get(c, None) for c in keep_cols}
    out["status"] = "completed"
    return out


def append_debug_history_to_iter_csv(result: Dict[str, Any], csv_path: str):
    debug_history = result.get("debug_history", [])
    if not debug_history:
        return

    df_new = pd.DataFrame(debug_history)
    df_new.insert(0, "instance", result.get("instance"))
    df_new.insert(1, "algorithm", result.get("algorithm"))
    df_new.insert(2, "beta", result.get("beta"))
    df_new.insert(3, "run_id", result.get("run_id"))
    df_new.insert(4, "seed", result.get("seed"))

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        all_cols = list(dict.fromkeys(list(df_old.columns) + list(df_new.columns)))
        df_old = df_old.reindex(columns=all_cols)
        df_new = df_new.reindex(columns=all_cols)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def _build_summary_pymoo(df: pd.DataFrame):
    required = {"instance", "algorithm", "beta"}
    if not required.issubset(df.columns):
        raise ValueError(f"_build_summary_pymoo missing required columns: {required - set(df.columns)}")

    agg_dict = {}

    def add_metric(col, ops):
        if col in df.columns:
            agg_dict[col] = ops

    add_metric("best_cost_in_archive", ["count", "mean", "std", "min", "median"])
    add_metric("best_ra_in_archive", ["mean", "std", "min", "median"])
    add_metric("runtime_sec", ["mean", "std", "min", "max", "median"])
    add_metric("archive_size", ["mean", "std", "max", "median"])
    add_metric("n_gen", ["mean", "std", "max", "median"])
    add_metric("ra_of_best_cost", ["mean", "std", "min", "median"])
    add_metric("cost_of_best_ra", ["mean", "std", "min", "median"])

    if len(agg_dict) == 0:
        return pd.DataFrame(columns=["instance", "algorithm", "beta"])

    summary = df.groupby(["instance", "algorithm", "beta"]).agg(agg_dict)
    summary.columns = ["_".join(col).strip() for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()

    return summary



def _worker_run_one_pymoo(task):
    (
        instance_path,
        instance_name,
        algorithm,
        beta,
        run_id,
        seed,
        kwargs,
        ev_params,
    ) = task

    try:
        config = ExperimentConfig(
            instance_path=instance_path,
            algorithm=algorithm,
            base_seed=seed - run_id,
            run_id=run_id,
            pop_size=kwargs["pop_size"],
            n_eval=kwargs["n_eval"],
            n_solutions=kwargs["n_solutions"],
            n_scenarios=kwargs["n_scenarios"],
            n_reduced_scenarios=kwargs["n_reduced_scenarios"],
            beta=beta,
            alpha=kwargs["alpha"],
            verbose=kwargs["verbose"],
            ra_safe=kwargs["ra_safe"],
            ra_risk=kwargs["ra_risk"],
            time_limit=kwargs["time_limit"],
        )

        rec_full = run_once(config, ev_params)

        rec_full["instance"] = rec_full.get("instance", instance_name)
        rec_full["instance_path"] = rec_full.get("instance_path", os.path.abspath(instance_path))
        rec_full["algorithm"] = str(rec_full.get("algorithm", algorithm)).lower()
        rec_full["beta"] = float(rec_full.get("beta", beta))
        rec_full["run_id"] = int(rec_full.get("run_id", run_id))
        rec_full["seed"] = int(rec_full.get("seed", seed))

        return {
            "ok": True,
            "instance": instance_name,
            "algorithm": algorithm,
            "beta": float(beta),
            "run_id": int(run_id),
            "seed": int(seed),
            "result": rec_full,
            "error": None,
        }

    except Exception as e:
        return {
            "ok": False,
            "instance": instance_name,
            "instance_path": os.path.abspath(instance_path),
            "algorithm": algorithm,
            "beta": float(beta),
            "run_id": int(run_id),
            "seed": int(seed),
            "result": None,
            "error": repr(e),
        }
    

def run_pymoo_experiments_resume_parallel(
    instance_source: str,
    algorithms: List[str],
    betas: List[float],
    n_runs: int,
    ev_params: Any,
    results_dir: Optional[str] = None,
    base_seed: int = 42,
    max_workers: int = 10,
    update_summary_each_run: bool = True,
    continue_on_error: bool = True,
    recursive: bool = True,
    instance_suffix: str = ".txt",

    pop_size: int = 50,
    n_eval: int = 50000,
    n_solutions: int = 20,
    n_scenarios: int = 100,
    n_reduced_scenarios: int = 10,
    alpha: float = 0.50,
    ra_safe: float = 0.10,
    ra_risk: float = 0.10,
    time_limit: str = "00:20:00",
    verbose: bool = False,
):
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    if results_dir is None:
        results_dir = os.path.join(os.getcwd(), "pymoo_experiment_results_parallel", run_tag)
    else:
        results_dir = os.path.join(results_dir, run_tag)

    os.makedirs(results_dir, exist_ok=True)

    progress_csv = os.path.join(results_dir, "progress_runs.csv")
    iter_csv = os.path.join(results_dir, "iter_log.csv")
    summary_live_csv = os.path.join(results_dir, "summary_live.csv")
    failed_csv = os.path.join(results_dir, "failed_runs.csv")
    final_xlsx = os.path.join(results_dir, "final_results.xlsx")
    detail_dir = os.path.join(results_dir, "run_details_json")
    os.makedirs(detail_dir, exist_ok=True)

    done = set()
    if os.path.exists(progress_csv):
        df_done = pd.read_csv(progress_csv)
        if {"instance", "algorithm", "beta", "run_id"}.issubset(df_done.columns):
            for _, row in df_done.iterrows():
                done.add(_make_key(row["instance"], row["algorithm"], row["beta"], row["run_id"]))
        print(f"[RESUME] Loaded progress: {progress_csv} | completed={len(done)}")
    else:
        print(f"[RESUME] No progress file. Will create: {progress_csv}")

    instance_list = _list_instances(instance_source=instance_source, recursive=recursive, suffix=instance_suffix)

    common_kwargs = {
        "pop_size": pop_size,
        "n_eval": n_eval,
        "n_solutions": n_solutions,
        "n_scenarios": n_scenarios,
        "n_reduced_scenarios": n_reduced_scenarios,
        "alpha": alpha,
        "ra_safe": ra_safe,
        "ra_risk": ra_risk,
        "time_limit": time_limit,
        "verbose": verbose,
    }

    tasks = []
    total_tasks = 0
    skipped = 0

    for instance_path in instance_list:
        instance_name = os.path.splitext(os.path.basename(instance_path))[0]
        for algorithm in algorithms:
            for beta in betas:
                for run_id in range(1, n_runs + 1):
                    total_tasks += 1
                    key = _make_key(instance_name, algorithm, beta, run_id)
                    if key in done:
                        skipped += 1
                        continue

                    seed = int(base_seed + 100000 * (algorithms.index(algorithm) + 1) + 1000 * round(float(beta) * 10) + (run_id - 1))
                    tasks.append((
                        instance_path,
                        instance_name,
                        algorithm,
                        float(beta),
                        int(run_id),
                        seed,
                        common_kwargs,
                        ev_params,
                    ))

    print(f"[INSTANCES] found={len(instance_list)}")
    print(f"[TASKS] total={total_tasks} | skipped(done)={skipped} | remaining={len(tasks)}")
    print(f"[PARALLEL] max_workers={max_workers}")

    n_ok = 0
    n_fail = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(_worker_run_one_pymoo, task): task for task in tasks}

        for i, future in enumerate(as_completed(future_to_task), 1):
            res = future.result()

            if res["ok"]:
                rec_full = res["result"]

                json_name = f"{rec_full['instance']}_{rec_full['algorithm']}_beta{rec_full['beta']:.3f}_run{rec_full['run_id']}.json".replace("/", "_")
                json_path = os.path.join(detail_dir, json_name)
                _save_json(json_path, rec_full)

                rec_flat = _extract_progress_record_pymoo(rec_full)
                _append_row_csv(progress_csv, rec_flat)
                append_debug_history_to_iter_csv(rec_full, iter_csv)

                done.add(_make_key(rec_flat["instance"], rec_flat["algorithm"], rec_flat["beta"], rec_flat["run_id"]))
                n_ok += 1

                print(
                    f"[OK {i}/{len(tasks)}] instance={rec_flat['instance']} algo={rec_flat['algorithm']} "
                    f"beta={rec_flat['beta']:.3f} run={rec_flat['run_id']} "
                    f"best_cost={rec_flat['best_cost_in_archive']:.6f} "
                    f"best_ra={rec_flat['best_ra_in_archive']:.6f}"
                )

                if update_summary_each_run:
                    df_all = pd.read_csv(progress_csv)
                    df_ok = df_all[df_all['status'] == 'completed'].copy() if 'status' in df_all.columns else df_all.copy()
                    _build_summary_pymoo(df_ok).to_csv(summary_live_csv, index=False)

            else:
                n_fail += 1
                fail_rec = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "instance": res["instance"],
                    "instance_path": res["instance_path"],
                    "algorithm": res["algorithm"],
                    "beta": float(res["beta"]),
                    "run_id": int(res["run_id"]),
                    "seed": int(res["seed"]),
                    "status": "failed",
                    "error": res["error"],
                }
                _append_row_csv(failed_csv, fail_rec)

                print(
                    f"[FAIL {i}/{len(tasks)}] instance={res['instance']} algo={res['algorithm']} "
                    f"beta={res['beta']:.3f} run={res['run_id']} error={res['error']}"
                )

                if not continue_on_error:
                    raise RuntimeError(res["error"])

    if os.path.exists(progress_csv):
        df_all = pd.read_csv(progress_csv)
        df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
        summary = _build_summary_pymoo(df_ok)

        with pd.ExcelWriter(final_xlsx, engine="openpyxl") as writer:
            df_all.to_excel(writer, sheet_name="runs", index=False)
            if os.path.exists(iter_csv):
                pd.read_csv(iter_csv).to_excel(writer, sheet_name="iter_log", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
            if os.path.exists(failed_csv):
                pd.read_csv(failed_csv).to_excel(writer, sheet_name="failed", index=False)

        print(f"\n[DONE] success={n_ok}, failed={n_fail}")
        print(f"[DONE] Progress CSV : {progress_csv}")
        print(f"[DONE] Iter Log CSV : {iter_csv}")
        print(f"[DONE] Detail JSONs  : {detail_dir}")
        print(f"[DONE] Final Excel   : {final_xlsx}")
        
# ============================================================
# 10. Example usage
# ============================================================

if __name__ == "__main__":
    ev_params = {
        "soc_max": 40.0,
        "vehicle_capacity": 650.0,
    }

    # config = ExperimentConfig(
    #     instance_path=r"D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)\c103_21.txt",
    #     algorithm="moead",
    #     run_id=0,
    #     pop_size=50,
    #     n_eval=50000,
    #     n_solutions=20,
    #     verbose=True,
    # )

    # result = run_once(config, ev_params)
    # print("best_cost_in_archive =", result["best_cost_in_archive"])
    # print("best_ra_in_archive =", result["best_ra_in_archive"])


    path = r"D:\02_Research\6_experimentResults\IGB_ACO_VNS\1"
    out_path = r"D:\02_Research\6_experimentResults\compare\cmopso_0401"

    run_pymoo_experiments_resume_parallel(
        instance_source=path,
        algorithms=["nsga3",], #"moead" "nsga2", "cmopso"
        betas=[0.2],
        n_runs=10,
        ev_params=ev_params,
        results_dir=out_path,
        base_seed=42,
        max_workers=10,
        recursive=True,
        pop_size=100,
        n_eval=50000,
        n_solutions=20,
        n_scenarios=100,
        n_reduced_scenarios=10,
        alpha=0.50,
        ra_safe=0.10,
        ra_risk=0.70,
        time_limit="00:20:00",   # 1200 秒
        verbose=False,
    )