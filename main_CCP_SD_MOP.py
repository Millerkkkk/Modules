from dataclasses import dataclass
import os
import random
import sys
import time
from typing import Optional

from scipy.stats import norm
import numpy as np
import pandas as pd
from datetime import datetime

from concurrent.futures import ProcessPoolExecutor, as_completed

from stochastic_demand import generate_normal_scenarios

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "./.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)



from params import ACOParams, GAParams
import copy

from GbPlanning_GA import plan_gb_order_ga
from CustomerPlanning_GA import plan_internal_gbs_order_ga

from GbPlanning import plan_gb_order
from CustomerPlanning import plan_internal_gbs_order

from Modules.EVRP.loader import load_problem      # 按你实际路径改
from Modules.Clustering.api import clustering               # 你重构后的统一入口
from Modules.Clustering.plot import plot_clusters
from Modules.GB.api import gb_clustering

from evaluate import Evaluator
from plot import plot_clusters_gbs, plot_clusters_gbs_with_gb_routes, plot_routes, plot_pareto_front, plot_search_evolution
from Modules.VNS.neighborhood import Neighborhood
from Modules.VNS.shaking import Shaker
from Modules.VNS.vnd import VNDRefiner


from time import perf_counter




NS = [
    "relocate_boundary",
    "merge_gb_intra_route",
    "relocate_gb_inter_route",
    "relocate_gb_intra_route",
]

NL = [
    "relocate_node_intra_gb",
    "reverse_node_intra_gb",
    "relocate_node_inter_gb",
    "swap_node_inter_gb",
    "relocate_node_inter_route",
    "relocate_boundary",
]


# NS = [
#     "gb_reverse",
#     "gb_relocate",
#     "gb_swap",
#     # "gb_inter_route_relocate",
#     # "gb_inter_route_exchange",
#     "boundary_relocate"
    
#     # "node_inter_route_shift_10",
#     # "node_inter_route_swap_11",
#     # "node_boundary_random_relocate_one"
# ]

# NL = [
#     "node_intra_gb_relocate",
#     "node_intra_gb_2opt",
#     "node_intra_gb_reverse",
#     # "node_intra_route_node_exchange",
#     "node_intra_route_relocate_across_any_gb",
#     # "node_boundary_random_relocate_one"
# ]

           

def max_expected_load(vehicle_capacity, beta, alpha):
    z = norm.ppf(alpha)
    return vehicle_capacity / (1 + z * beta)



# ---------- helpers ----------
def pack_solution(routes_by_clusters, cend):
        return {
            "routes_by_clusters": copy.deepcopy(routes_by_clusters),
            "cost": float(cend["total_cost"]),
            "ra": float(cend["total_ra"]),
            "cend": cend,
        }

def dominates(a, b, eps=1e-9):
    no_worse = (a["cost"] <= b["cost"] + eps) and (a["ra"] <= b["ra"] + eps)
    strictly_better = (a["cost"] < b["cost"] - eps) or (a["ra"] < b["ra"] - eps)
    return no_worse and strictly_better

def same_point(a, b, eps=1e-9):
    return abs(a["cost"] - b["cost"]) <= eps and abs(a["ra"] - b["ra"]) <= eps

def crowding_distance(pop, front):
    """对某个 front(索引列表)计算 crowding distance，返回 dict idx->dist"""
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            dist[i] = float("inf")
        return dist

    # 两个目标：cost, ra
    for key in ["cost", "ra"]:
        front_sorted = sorted(front, key=lambda i: pop[i][key])
        dist[front_sorted[0]] = float("inf")
        dist[front_sorted[-1]] = float("inf")

        min_v = pop[front_sorted[0]][key]
        max_v = pop[front_sorted[-1]][key]
        denom = (max_v - min_v) if (max_v - min_v) > 1e-12 else 1e-12

        for k in range(1, len(front_sorted) - 1):
            prev_v = pop[front_sorted[k - 1]][key]
            next_v = pop[front_sorted[k + 1]][key]
            dist[front_sorted[k]] += (next_v - prev_v) / denom

    return dist

def pareto_insert_strict(archive, cand, eps=1e-9):
    for s in archive:
        if dominates(s, cand, eps) or same_point(s, cand, eps):
            return archive, False

    new_archive = [s for s in archive if not dominates(cand, s, eps)]
    new_archive.append(cand)
    return new_archive, True

def truncate_by_crowding(arch, max_size):
    if len(arch) <= max_size:
        return arch

    front = list(range(len(arch)))
    cd = crowding_distance(arch, front)  # you already have this

    # cd might be dict-like or list-like; normalize accessor
    def cd_val(i):
        return cd[i] if isinstance(cd, (list, tuple, np.ndarray)) else cd.get(i, 0.0)

    keep_idx = sorted(front, key=lambda i: cd_val(i), reverse=True)[:max_size]
    return [arch[i] for i in keep_idx]

def pareto_set_insert(pareto_set, cand, eps=1e-9):
    for s in pareto_set:
        if dominates(s, cand, eps) or same_point(s, cand, eps):
            return pareto_set

    new_set = [s for s in pareto_set if not dominates(cand, s, eps)]
    new_set.append(cand)
    return new_set


def pick_work_solution(arch, topk=10):
    if not arch:
        return None
    # 先按 cost 排序取前 topk
    cand = sorted(arch, key=lambda s: s["cost"])[:min(topk, len(arch))]

    # 在 cand 里选 crowding 最大的（需要你已有 crowding_distance）
    idxs = list(range(len(cand)))
    cd = crowding_distance(cand, idxs)
    def cd_val(i): return cd[i] if isinstance(cd, (list, tuple, np.ndarray)) else cd.get(i, 0.0)

    return cand[max(idxs, key=cd_val)]


def pick_work_solution1(arch, rng=None):
    """
    在 archive 中：
    - 以 50% 概率选 cost 最小的解
    - 以 50% 概率选 ra 最小的解
    """
    if not arch:
        return None

    rng = rng if rng is not None else np.random.default_rng()

    best_cost = min(arch, key=lambda s: s["cost"])
    best_ra   = min(arch, key=lambda s: s["ra"])

    # 随机选一个极端
    if rng.random() < 0.5:
        return best_cost
    else:
        return best_ra
            
def pick_work_solution2(arch, eps=1e-12):
    if not arch:
        return None

    costs = [s["cost"] for s in arch]
    ras   = [s["ra"]   for s in arch]
    cmin, cmax = min(costs), max(costs)
    rmin, rmax = min(ras),   max(ras)

    def norm(x, a, b):
        return 0.0 if abs(b - a) < eps else (x - a) / (b - a)

    def score(s):
        c = norm(s["cost"], cmin, cmax)
        r = norm(s["ra"],   rmin, rmax)
        return (c*c + r*r) ** 0.5

    return min(arch, key=score)

def pick_work_random(arch, rng=None):
    """
    从 archive 中随机选一个 work
    """
    if not arch:
        return None

    rng = rng if rng is not None else np.random.default_rng()
    idx = int(rng.integers(len(arch)))
    return arch[idx]

def pick_work_crowding_prob(arch, rng=None, eps=1e-12):
    """
    按 crowding distance 的大小作为权重随机选解
    crowding 越大，被选中的概率越高
    """
    if not arch:
        return None
    if len(arch) == 1:
        return arch[0]

    rng = rng if rng is not None else np.random.default_rng()

    idxs = list(range(len(arch)))
    cd = crowding_distance(arch, idxs)

    vals = []
    for i in idxs:
        if isinstance(cd, (list, tuple, np.ndarray)):
            v = float(cd[i])
        else:
            v = float(cd.get(i, 0.0))

        # 防止 inf / nan / 负数
        if not np.isfinite(v) or v < 0:
            v = 0.0
        vals.append(v)

    vals = np.asarray(vals, dtype=float)

    # 如果全是 0，就退化成均匀随机
    if vals.sum() < eps:
        idx = int(rng.integers(len(arch)))
        return arch[idx]

    probs = vals / vals.sum()
    idx = int(rng.choice(len(arch), p=probs))
    return arch[idx]

def build_grid_cells(arch, n_grid=10, eps=1e-12):
    if not arch:
        return {}

    costs = np.asarray([s["cost"] for s in arch], dtype=float)
    ras   = np.asarray([s["ra"]   for s in arch], dtype=float)

    cmin, cmax = float(costs.min()), float(costs.max())
    rmin, rmax = float(ras.min()),   float(ras.max())

    def to_bin(x, a, b):
        if abs(b - a) < eps:
            return 0
        z = (x - a) / (b - a)
        z = min(max(z, 0.0), 1.0)
        k = int(z * n_grid)
        if k >= n_grid:
            k = n_grid - 1
        return k

    cells = {}
    for i, s in enumerate(arch):
        cx = to_bin(float(s["cost"]), cmin, cmax)
        ry = to_bin(float(s["ra"]),   rmin, rmax)
        cell = (cx, ry)
        cells.setdefault(cell, []).append(i)

    return cells

def pick_work_grid(arch, rng=None, n_grid=10, select_pressure=2.0, eps=1e-12):
    if not arch:
        return None
    if len(arch) == 1:
        return arch[0]

    rng = rng if rng is not None else np.random.default_rng()

    cells = build_grid_cells(arch, n_grid=n_grid, eps=eps)
    cell_keys = list(cells.keys())

    densities = np.asarray([len(cells[k]) for k in cell_keys], dtype=float)
    weights = 1.0 / np.power(np.maximum(densities, eps), select_pressure)
    probs = weights / weights.sum()

    chosen_cell = cell_keys[int(rng.choice(len(cell_keys), p=probs))]
    chosen_idx = int(rng.choice(cells[chosen_cell]))

    return arch[chosen_idx]




def run_once_ccp(
    instance_path: str,
    beta: float,
    alpha: float,
    run_id: int,
    seed: int,
    verbose: int = 1,
    log_every: int = 20,
    K: int = 1,
    A_MAX: int = 80,
    max_iter: int = 2000,
    patience: int = 50,
    eps = 1e-9
):
    """
    单次实验：多目标 VNS + VND（GBACOPlanner 风格）

    风格特征
    --------
    1. 每个邻域层级 k 采样 K 个候选
    2. 当前 k 下维护一个局部非支配集 best_local_set
    3. 若 best_local_set 中存在支配 base 的点，则接受其中一个并令 k=1
    4. 每轮结束后，从全局 Pareto archive 中重新选择 work 解

    目标
    ----
    - minimize total_cost
    - minimize total_ra

    参数
    ----
    instance_path : str
    beta          : float
    run_id        : int
    seed          : int
    return_details: bool
        False -> 只返回适合写 Excel 的标量字段
        True  -> 额外返回 archive / work_history / final_best / debug_log / cand_points 等
    verbose       : int
        0 -> 不打印
        1 -> 每 log_every 轮打印摘要
        2 -> 每轮打印摘要
        3 -> 每个 candidate 都打印
    log_every     : int
        verbose=1 时的打印间隔
    K             : int
        每个邻域层级 k 采样的候选数
    A_MAX         : int
        archive 最大大小
    max_iter      : int
        外层最大迭代次数
    patience      : int
        archive 连续多少轮不变就早停
    """
    
    timers = {
        "init": 0.0,
        "shake": 0.0,
        "refine": 0.0,
        "archive": 0.0,
        "pick": 0.0,
        "outer_iter": 0.0,
    }
    counts = {
        "cand": 0,
        "refine_calls": 0,
    }

    # =============================
    # 0. 初始化
    # =============================
    t0_all = perf_counter()
    t = perf_counter()

    instance_name = os.path.splitext(os.path.basename(instance_path))[0]

    ev_params = {
        "soc_max": 40.0,
        "vehicle_capacity": 650.0,
    }

    problem = load_problem(instance_path, ev_params)
    evaluator = Evaluator(problem, ra_safe=0.1, ra_risk=0.7)

    # =============================
    # 1. 计算载重最大期望值
    # =============================
    expected_vehicle_capacity = max_expected_load(problem.vehicle_capacity, beta, alpha)

    # =============================
    # 2. clustering
    # =============================
    clusters = clustering(
        problem,
        method="kmeans",
        feature="location",
        vehicle_capacity=expected_vehicle_capacity,
        random_state=seed,   # 原来是 1，改成 seed
    )

    depot_coords = np.asarray(
        [(problem.nodes[i].x, problem.nodes[i].y) for i in problem.depots],
        dtype=float,
    )

    # =============================
    # 3. 初始化：GB + GA
    # =============================
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
            split_k=0.4,
            merge=True,
            random_state=seed + cid,   # 原来 None，改成可复现实验
        )

        gbs_customer_id = [[customer_ids[j] for j in cl] for cl in gbs_info.gbs]

        # ===== 规划 GB 之间路径（GA）=====
        params_gb = GAParams(
            pop_size=30,
            num_gen=100,
            tournament_k=3,
            crossover_rate=0.9,
            mutation_rate=0.2,
            inversion_rate=0.7,
            elite_size=2,
            seed=seed + 1000 + cid,   # 原来 0
        )
        gb_order_result = plan_gb_order_ga(
            gbs_info.centers,
            depot_coords,
            params_gb,
        )

        # ===== 规划 GB 内客户路径（GA）=====
        params_customer = GAParams(
            pop_size=60,
            num_gen=100,
            tournament_k=3,
            crossover_rate=0.9,
            mutation_rate=0.3,
            inversion_rate=0.7,
            elite_size=2,
            seed=seed + 2000 + cid,   # 原来 0
        )
        routes = plan_internal_gbs_order_ga(
            problem,
            gb_order_result.order,
            gbs_info.centers,
            gbs_customer_id,
            params=params_customer,
        )


        # params_GB = ACOParams(num_ants=10, num_iter=60)
        # gb_order_result = plan_gb_order(gbs_info.centers, depot_coords, params_GB)

        # params_customer = ACOParams(num_ants=20, num_iter=100, seed=seed + 2000 + cid)
        # routes = plan_internal_gbs_order(problem, gb_order_result.order, gbs_info.centers, gbs_customer_id,
        #     params=params_customer,
        #     )
 

        routes_clusters[cid] = routes

    # =============================
    # 4. 初始解评估
    # =============================
    init_eval, routes_clusters = evaluator.evaluate_init(routes_clusters)
    init_sol = pack_solution(routes_clusters, init_eval)

    archive = []
    archive, _ = pareto_insert_strict(archive, init_sol, eps=eps)
    archive = truncate_by_crowding(archive, max_size=A_MAX)

    work = pick_work_grid(archive)
    if work is None:
        raise RuntimeError("Archive is empty after initialization.")

    if verbose >= 1:
        print(
            f"[Init] instance={instance_name} beta={beta:.3f} seed={seed} "
            f"cost={work['cost']:.6f} ra={work['ra']:.6f} archive={len(archive)}"
        )

    work_history = [(0, float(work["cost"]), float(work["ra"]))]
    cand_points = [(float(init_sol["cost"]), float(init_sol["ra"]))]

    # 统一统计口径：把初始解也算一个 candidate
    counts["cand"] = 1

    timers["init"] += perf_counter() - t

    # =============================
    # 5. Multi-objective VNS + VND
    # =============================
    rng = np.random.default_rng(seed)

    nb = Neighborhood(
        problem=problem,
        removal_ratio=0.30,
        alpha_boundary=0.50,
        min_gb_len=2,
        rng=rng,
        evaluator=None,   # 当前最小修改，先不动
    )

    shaker = Shaker(nb, NS)

    vnd = VNDRefiner(
        nb,
        evaluator=lambda routes: evaluator.evaluate_clusters(routes),
        eval_proxy=None,
        dominates_fn=dominates,
        rng=rng,
        eps=eps,
    )

    no_improve = 0
    k_max = len(NS)

    for it in range(1, max_iter + 1):
        archive_changed = False
        base = copy.deepcopy(work)

        iter_t0 = perf_counter()

        k = 1
        while k <= k_max:
            best_local_set = []

            # ---------------------------------
            # 在当前 k 下，采样 K 个候选
            # ---------------------------------
            for kk in range(K):
                base_routes = copy.deepcopy(base["routes_by_clusters"])

                # 1) shaking
                t_shake = perf_counter()
                cand_routes = shaker.shake(base_routes, k=k, cid=None)
                timers["shake"] += perf_counter() - t_shake

                # 2) VND refine
                t_refine = perf_counter()
                cand_routes, cand_cend, cand_info = vnd.refine(
                    cand_routes,
                    NL=NL,
                    cid=None,
                    tries_per_op=2,
                    max_steps=30,
                )
                timers["refine"] += perf_counter() - t_refine
                counts["refine_calls"] += 1

                # 3) pack candidate
                cand = pack_solution(cand_routes, cand_cend)
                cand_points.append((float(cand["cost"]), float(cand["ra"])))
                counts["cand"] += 1

                # 4) 更新全局 archive
                t_archive = perf_counter()
                old_points = {(float(s["cost"]), float(s["ra"])) for s in archive}

                archive, _ = pareto_insert_strict(archive, cand, eps=eps)
                archive = truncate_by_crowding(archive, max_size=A_MAX)

                new_points = {(float(s["cost"]), float(s["ra"])) for s in archive}
                if new_points != old_points:
                    archive_changed = True

                timers["archive"] += perf_counter() - t_archive

                # 5) 更新 Nk 内 Pareto 集
                best_local_set = pareto_set_insert(best_local_set, cand, eps=eps)

            # ---------------------------------
            # 在 best_local_set 中判断是否有点支配 base
            # ---------------------------------
            improvers = [s for s in best_local_set if dominates(s, base, eps)]

            if improvers:
                # 原来只按 cost 选，偏置太强
                # 最小修改：随机选一个支配 base 的候选

                base = min(improvers, key=lambda s: (s["cost"], s["ra"]))
                # base = min(improvers, key=lambda s: s["cost"])
                k = 1
            else:
                k += 1

        # ---------------------------------
        # 一轮结束：从 archive 挑下一个 work
        # ---------------------------------
        t_pick = perf_counter()
        work = pick_work_grid(archive)
        timers["pick"] += perf_counter() - t_pick

        if work is None:
            raise RuntimeError("Archive became empty during search.")

        work_history.append((it, float(work["cost"]), float(work["ra"])))

        if verbose >= 1 and (it % log_every) == 0:
            cur_best_cost = min(s["cost"] for s in archive)
            cur_best_ra = min(s["ra"] for s in archive)
            print(
                f"[Iter {it}] archive={len(archive)} "
                f"min_cost_in_arch={cur_best_cost:.3f} "
                f"min_ra_in_arch={cur_best_ra:.6f} "
                f"work=(cost={work['cost']:.3f}, ra={work['ra']:.6f})"
            )

        # early stop: 基于 archive 是否变化
        if archive_changed:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose >= 1:
                    print(f"[Stop] no improvement for {patience} outer iterations.")
                break

        iter_time = perf_counter() - iter_t0
        timers["outer_iter"] += iter_time


    # =============================
    # 6. 收尾统计
    # =============================
    runtime = perf_counter() - t0_all

    t_pick = perf_counter()
    final_best = pick_work_grid(archive)
    timers["pick"] += perf_counter() - t_pick
    if final_best is None:
        raise RuntimeError("Archive is empty at the end of search.")

    # avg_refine = timers["refine"] / max(counts["refine_calls"], 1)
    # avg_cand = timers["outer_iter"] / max(counts["cand"], 1)

    # print("\n[Profile Summary]")
    # print(f"  init      : {timers['init']:.4f} s")
    # print(f"  shake     : {timers['shake']:.4f} s")
    # print(f"  refine    : {timers['refine']:.4f} s")
    # print(f"  archive   : {timers['archive']:.4f} s")
    # print(f"  pick      : {timers['pick']:.4f} s")
    # print(f"  outer_iter: {timers['outer_iter']:.4f} s")
    # print(f"  runtime   : {runtime:.4f} s")

    # print("\n[Counts]")
    # print(f"  candidates    : {counts['cand']}")
    # print(f"  refine_calls  : {counts['refine_calls']}")
    # print(f"  avg_refine    : {avg_refine:.6f} s/call")
    # print(f"  avg_outer/cand: {avg_cand:.6f} s/candidate")

    min_cost_sol = min(archive, key=lambda s: s["cost"])
    min_ra_sol = min(archive, key=lambda s: s["ra"])

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "instance": instance_name,
        "beta": float(beta),
        "alpha": float(alpha),
        "run_id": int(run_id),
        "seed": int(seed),

        "init_cost": float(init_sol["cost"]),
        "init_ra": float(init_sol["ra"]),

        "final_cost": float(final_best["cost"]),
        "final_ra": float(final_best["ra"]),

        "best_cost_in_archive": float(min_cost_sol["cost"]),
        "ra_of_best_cost": float(min_cost_sol["ra"]),

        "best_ra_in_archive": float(min_ra_sol["ra"]),
        "cost_of_best_ra": float(min_ra_sol["cost"]),

        "archive_size": int(len(archive)),
        "n_candidates": int(counts["cand"]),
        "n_iterations": int(len(work_history) - 1),
        "runtime_sec": float(runtime),

        "pareto_points": [(float(sol["cost"]), float(sol["ra"])) for sol in archive],
        "archive": archive,
        "work_history": work_history,
        "final_best": final_best,
        "min_cost_sol": min_cost_sol,
        "min_ra_sol": min_ra_sol,
        "cand_points": cand_points,
        "clusters": clusters,
    }

    return result









import os
import pickle
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


# =========================
# 工具函数：列举 instances
# =========================
def _list_instances(instance_source: str):
    if os.path.isfile(instance_source):
        return [instance_source]
    return sorted(
        os.path.join(instance_source, f)
        for f in os.listdir(instance_source)
        if f.lower().endswith(".txt")
    )


# =========================
# 工具函数：任务唯一 key
# =========================
def _make_key(instance: str, beta: float, run_id: int):
    return (str(instance), round(float(beta), 6), int(run_id))


# =========================
# 工具函数：安全追加一行 CSV
# =========================
def _append_row_csv(path: str, record: dict):
    df_row = pd.DataFrame([record])
    write_header = not os.path.exists(path)
    df_row.to_csv(path, mode="a", header=write_header, index=False)


# =========================
# 工具函数：安全追加多行 CSV
# =========================
def _append_rows_csv(path: str, rows: list[dict]):
    if not rows:
        return
    df_rows = pd.DataFrame(rows)
    write_header = not os.path.exists(path)
    df_rows.to_csv(path, mode="a", header=write_header, index=False)


# =========================
# 工具函数：从 progress 构建 summary
# =========================
def _build_summary(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame()

    agg_dict = {
        "final_cost": ["count", "mean", "std", "min"],
        "final_ra": ["mean", "std", "min"],
        "best_cost_in_archive": ["mean", "std", "min"],
        "best_ra_in_archive": ["mean", "std", "min"],
        "archive_size": ["mean", "std", "max"],
        "runtime_sec": ["mean", "std"],
    }

    if "t_stage1" in df.columns:
        agg_dict["t_stage1"] = ["mean", "std"]
    if "t_stage2" in df.columns:
        agg_dict["t_stage2"] = ["mean", "std"]

    summary = (
        df.groupby(["instance", "beta"])
          .agg(agg_dict)
    )

    summary.columns = ["_".join(col).strip() for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()

    if "t_stage2_mean" in summary.columns and "runtime_sec_mean" in summary.columns:
        summary["stage2_ratio_mean"] = summary["t_stage2_mean"] / summary["runtime_sec_mean"]

    return summary


# =========================
# 工具函数：保存详细对象为 PKL
# =========================
def _save_run_details_pkl(
    details_dir: str,
    instance_name: str,
    beta: float,
    run_id: int,
    detail_rec: dict,
):
    os.makedirs(details_dir, exist_ok=True)
    fname = f"{instance_name}_beta{beta:.6f}_run{run_id}.pkl"
    path = os.path.join(details_dir, fname)
    with open(path, "wb") as f:
        pickle.dump(detail_rec, f)
    return path


# =========================
# 工具函数：把 archive 摊平成多行
# =========================
def _make_archive_rows(rec: dict):
    archive = rec.get("archive", [])
    final_best = rec.get("final_best", None)
    min_cost_sol = rec.get("min_cost_sol", None)
    min_ra_sol = rec.get("min_ra_sol", None)

    rows = []
    for i, s in enumerate(archive):
        row = {
            "timestamp": rec.get("timestamp"),
            "instance": rec.get("instance"),
            "beta": float(rec.get("beta")),
            "run_id": int(rec.get("run_id")),
            "seed": int(rec.get("seed")),
            "sol_id": int(i),
            "cost": float(s["cost"]),
            "ra": float(s["ra"]),
            "is_final_best": bool(same_point(s, final_best)) if final_best is not None else False,
            "is_min_cost": bool(same_point(s, min_cost_sol)) if min_cost_sol is not None else False,
            "is_min_ra": bool(same_point(s, min_ra_sol)) if min_ra_sol is not None else False,
        }
        rows.append(row)
    return rows


# =========================
# 工具函数：拆分 summary / detail
# =========================
def _split_run_record(rec: dict):
    detail_keys = {
        "archive",
        "work_history",
        "final_best",
        "min_cost_sol",
        "min_ra_sol",
        "problem",
        "scenarios",
        "cand_points",
        "debug_log",
        "clusters",
    }

    detail_rec = {k: rec[k] for k in rec if k in detail_keys}
    summary_rec = {k: rec[k] for k in rec if k not in detail_keys}
    return summary_rec, detail_rec


# =========================
# 终极稳定版：断点续跑 + 实时落盘 + 保存 Pareto 解集 + 详细对象
# =========================
def run_experiments_resume_ultimate(
    BETAS,
    ALPHAS,
    N_RUNS,
    instance_source: str,
    results_dir: Optional[str] = None,
    base_seed: int = 42,
    update_summary_each_run: bool = True,
    continue_on_error: bool = True,
):
    """
    终极稳定版：
    - progress_runs.csv     : 每个 run 的摘要，一行一条
    - summary_live.csv      : 每个 run 后更新均值
    - failed_runs.csv       : 失败记录
    - pareto_archive.csv    : 每个 Pareto 点一行
    - run_details/*.pkl     : 每个 run 的完整详细对象
    - final_results.xlsx    : 导出 runs + summary + pareto_archive
    - 支持断点续跑，自动跳过已完成项
    """

    # -------------------------
    # 创建输出目录（每次实验自动建时间文件夹）
    # -------------------------
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    if results_dir is None:
        results_dir = os.path.join(os.getcwd(), "beta_saa_results", run_tag)
    else:
        results_dir = os.path.join(results_dir, run_tag)

    os.makedirs(results_dir, exist_ok=True)

    progress_csv = os.path.join(results_dir, "progress_runs.csv")
    summary_live_csv = os.path.join(results_dir, "summary_live.csv")
    failed_csv = os.path.join(results_dir, "failed_runs.csv")
    pareto_csv = os.path.join(results_dir, "pareto_archive.csv")
    final_xlsx = os.path.join(results_dir, "final_results.xlsx")
    details_dir = os.path.join(results_dir, "run_details")

    # -------------------------
    # 0) 读取已完成记录
    # -------------------------
    done = set()
    if os.path.exists(progress_csv):
        df_done = pd.read_csv(progress_csv)
        if {"instance", "beta", "run_id"}.issubset(df_done.columns):
            for _, row in df_done.iterrows():
                done.add(_make_key(row["instance"], row["beta"], row["run_id"]))
        print(f"[RESUME] Loaded progress: {progress_csv} | completed={len(done)}")
    else:
        print(f"[RESUME] No progress file. Will create: {progress_csv}")

    # -------------------------
    # 1) 生成任务列表
    # -------------------------
    instance_list = _list_instances(instance_source)

    tasks = []
    total_tasks = 0
    skipped = 0

    for instance_path in instance_list:
        instance_name = os.path.splitext(os.path.basename(instance_path))[0]
        for beta in BETAS:
            for run_id in range(1, N_RUNS + 1):
                total_tasks += 1
                key = _make_key(instance_name, beta, run_id)

                if key in done:
                    skipped += 1
                    continue

                seed = base_seed + 1000 * int(round(beta * 10)) + (run_id - 1)
                tasks.append((instance_path, instance_name, beta, run_id, seed))

    print(f"[TASKS] total={total_tasks} | skipped(done)={skipped} | remaining={len(tasks)}")

    # -------------------------
    # 如果没有剩余任务，也刷新最终输出
    # -------------------------
    if len(tasks) == 0 and os.path.exists(progress_csv):
        df_all = pd.read_csv(progress_csv)
        df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
        summary = _build_summary(df_ok)

        if os.path.exists(pareto_csv):
            df_pareto = pd.read_csv(pareto_csv)
        else:
            df_pareto = pd.DataFrame()

        with pd.ExcelWriter(final_xlsx, engine="openpyxl") as writer:
            df_all.to_excel(writer, sheet_name="runs", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
            df_pareto.to_excel(writer, sheet_name="pareto_archive", index=False)

        print(f"[DONE] Nothing to run. Final Excel refreshed: {final_xlsx}")
        return

    # -------------------------
    # 2) 执行任务：每次 run 结束立即落盘
    # -------------------------
    for idx, (instance_path, instance_name, beta, run_id, seed) in enumerate(tasks, 1):
        print(f"\n=== [{idx}/{len(tasks)}] instance={instance_name} beta={beta:.1f} run={run_id}/{N_RUNS} seed={seed} ===")

        try:
            rec = run_once(
                instance_path=instance_path,
                beta=beta,
                run_id=run_id,
                seed=seed,
                return_details=True,
            )

            # 拆分摘要和详细对象
            summary_rec, detail_rec = _split_run_record(rec)

            # 保存详细对象
            detail_path = _save_run_details_pkl(
                details_dir=details_dir,
                instance_name=instance_name,
                beta=beta,
                run_id=run_id,
                detail_rec=detail_rec,
            )

            # 补齐摘要关键字段
            summary_rec["instance"] = summary_rec.get("instance", instance_name)
            summary_rec["beta"] = float(summary_rec.get("beta", beta))
            summary_rec["run_id"] = int(summary_rec.get("run_id", run_id))
            summary_rec["seed"] = int(summary_rec.get("seed", seed))
            summary_rec["status"] = "completed"
            summary_rec["details_path"] = detail_path

            # 先保存 progress
            _append_row_csv(progress_csv, summary_rec)

            # 再保存 Pareto 点
            archive_rows = _make_archive_rows(rec)
            _append_rows_csv(pareto_csv, archive_rows)

            # 更新 done
            done.add(_make_key(summary_rec["instance"], summary_rec["beta"], summary_rec["run_id"]))

            # 可选：实时更新 summary
            if update_summary_each_run:
                df_all = pd.read_csv(progress_csv)
                df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
                summary = _build_summary(df_ok)
                summary.to_csv(summary_live_csv, index=False)

        except Exception as e:
            err = repr(e)
            print(f"[ERROR] instance={instance_name} beta={beta} run={run_id} failed: {err}")

            fail_rec = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "instance": instance_name,
                "beta": float(beta),
                "run_id": int(run_id),
                "seed": int(seed),
                "status": "failed",
                "error": err,
            }
            _append_row_csv(failed_csv, fail_rec)

            if not continue_on_error:
                raise

            continue

    # -------------------------
    # 3) 跑完后生成最终 Excel
    # -------------------------
    if os.path.exists(progress_csv):
        df_all = pd.read_csv(progress_csv)
        df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
        summary = _build_summary(df_ok)

        if os.path.exists(pareto_csv):
            df_pareto = pd.read_csv(pareto_csv)
        else:
            df_pareto = pd.DataFrame()

        with pd.ExcelWriter(final_xlsx, engine="openpyxl") as writer:
            df_all.to_excel(writer, sheet_name="runs", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
            df_pareto.to_excel(writer, sheet_name="pareto_archive", index=False)

        print(f"\n[DONE] Progress CSV : {progress_csv}")
        if update_summary_each_run:
            print(f"[DONE] Live Summary : {summary_live_csv}")
        if os.path.exists(failed_csv):
            print(f"[DONE] Failed Runs  : {failed_csv}")
        if os.path.exists(pareto_csv):
            print(f"[DONE] Pareto CSV   : {pareto_csv}")
        print(f"[DONE] Details Dir   : {details_dir}")
        print(f"[DONE] Final Excel   : {final_xlsx}")

    else:
        print("[DONE] No progress file found; nothing to export.")







def _run_one_task(args):
    instance_path, instance_name, beta, alpha, run_id, seed = args
    try:
        rec = run_once(
            instance_path=instance_path,
            beta=beta,
            alpha=alpha,
            run_id=run_id,
            seed=seed,
        )
        return {
            "ok": True,
            "instance_name": instance_name,
            "beta": beta,
            "run_id": run_id,
            "seed": seed,
            "record": rec,
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "instance_name": instance_name,
            "beta": beta,
            "run_id": run_id,
            "seed": seed,
            "record": None,
            "error": repr(e),
        }


def run_experiments_resume_ultimate_10_worker(
    BETAS,
    N_RUNS,
    instance_source: str,
    results_dir: Optional[str] = None,
    base_seed: int = 42,
    update_summary_each_run: bool = True,
    continue_on_error: bool = True,
):
    """
    终极稳定版：
    - progress_runs.csv     : 每个 run 的摘要，一行一条
    - summary_live.csv      : 每个 run 后更新均值
    - failed_runs.csv       : 失败记录
    - pareto_archive.csv    : 每个 Pareto 点一行
    - run_details/*.pkl     : 每个 run 的完整详细对象
    - final_results.xlsx    : 导出 runs + summary + pareto_archive
    - 支持断点续跑，自动跳过已完成项
    """

    # -------------------------
    # 创建输出目录（每次实验自动建时间文件夹）
    # -------------------------
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    if results_dir is None:
        results_dir = os.path.join(os.getcwd(), "beta_saa_results", run_tag)
    else:
        results_dir = os.path.join(results_dir, run_tag)

    os.makedirs(results_dir, exist_ok=True)

    progress_csv = os.path.join(results_dir, "progress_runs.csv")
    summary_live_csv = os.path.join(results_dir, "summary_live.csv")
    failed_csv = os.path.join(results_dir, "failed_runs.csv")
    pareto_csv = os.path.join(results_dir, "pareto_archive.csv")
    final_xlsx = os.path.join(results_dir, "final_results.xlsx")
    details_dir = os.path.join(results_dir, "run_details")

    # -------------------------
    # 0) 读取已完成记录
    # -------------------------
    done = set()
    if os.path.exists(progress_csv):
        df_done = pd.read_csv(progress_csv)
        if {"instance", "beta", "run_id"}.issubset(df_done.columns):
            for _, row in df_done.iterrows():
                done.add(_make_key(row["instance"], row["beta"], row["run_id"]))
        print(f"[RESUME] Loaded progress: {progress_csv} | completed={len(done)}")
    else:
        print(f"[RESUME] No progress file. Will create: {progress_csv}")

    # -------------------------
    # 1) 生成任务列表
    # -------------------------
    instance_list = _list_instances(instance_source)

    tasks = []
    total_tasks = 0
    skipped = 0

    for instance_path in instance_list:
        instance_name = os.path.splitext(os.path.basename(instance_path))[0]
        for beta in BETAS:
            for run_id in range(1, N_RUNS + 1):
                total_tasks += 1
                key = _make_key(instance_name, beta, run_id)

                if key in done:
                    skipped += 1
                    continue

                seed = base_seed + 1000 * int(round(beta * 10)) + (run_id - 1)
                tasks.append((instance_path, instance_name, beta, run_id, seed))

    print(f"[TASKS] total={total_tasks} | skipped(done)={skipped} | remaining={len(tasks)}")

    # -------------------------
    # 如果没有剩余任务，也刷新最终输出
    # -------------------------
    if len(tasks) == 0 and os.path.exists(progress_csv):
        df_all = pd.read_csv(progress_csv)
        df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
        summary = _build_summary(df_ok)

        if os.path.exists(pareto_csv):
            df_pareto = pd.read_csv(pareto_csv)
        else:
            df_pareto = pd.DataFrame()

        with pd.ExcelWriter(final_xlsx, engine="openpyxl") as writer:
            df_all.to_excel(writer, sheet_name="runs", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
            df_pareto.to_excel(writer, sheet_name="pareto_archive", index=False)

        print(f"[DONE] Nothing to run. Final Excel refreshed: {final_xlsx}")
        return

    # -------------------------
    # 2) 并行执行任务：10 workers
    # -------------------------
    max_workers = 10

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_run_one_task, task): task
            for task in tasks
        }

        for idx, future in enumerate(as_completed(future_to_task), 1):
            instance_path, instance_name, beta, run_id, seed = future_to_task[future]

            print(
                f"\n=== [{idx}/{len(tasks)} DONE] "
                f"instance={instance_name} beta={beta:.1f} run={run_id}/{N_RUNS} seed={seed} ==="
            )

            try:
                out = future.result()
            except Exception as e:
                err = repr(e)
                print(f"[ERROR] instance={instance_name} beta={beta} run={run_id} failed: {err}")

                fail_rec = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "instance": instance_name,
                    "beta": float(beta),
                    "run_id": int(run_id),
                    "seed": int(seed),
                    "status": "failed",
                    "error": err,
                }
                _append_row_csv(failed_csv, fail_rec)

                if not continue_on_error:
                    raise
                continue

            if out["ok"]:
                rec = out["record"]

                summary_rec, detail_rec = _split_run_record(rec)

                detail_path = _save_run_details_pkl(
                    details_dir=details_dir,
                    instance_name=instance_name,
                    beta=beta,
                    run_id=run_id,
                    detail_rec=detail_rec,
                )

                summary_rec["instance"] = summary_rec.get("instance", instance_name)
                summary_rec["beta"] = float(summary_rec.get("beta", beta))
                summary_rec["run_id"] = int(summary_rec.get("run_id", run_id))
                summary_rec["seed"] = int(summary_rec.get("seed", seed))
                summary_rec["status"] = "completed"
                summary_rec["details_path"] = detail_path

                _append_row_csv(progress_csv, summary_rec)

                archive_rows = _make_archive_rows(rec)
                _append_rows_csv(pareto_csv, archive_rows)

                done.add(_make_key(summary_rec["instance"], summary_rec["beta"], summary_rec["run_id"]))

                if update_summary_each_run:
                    df_all = pd.read_csv(progress_csv)
                    df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
                    summary = _build_summary(df_ok)
                    summary.to_csv(summary_live_csv, index=False)

            else:
                err = out["error"]
                print(f"[ERROR] instance={instance_name} beta={beta} run={run_id} failed: {err}")

                fail_rec = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "instance": instance_name,
                    "beta": float(beta),
                    "run_id": int(run_id),
                    "seed": int(seed),
                    "status": "failed",
                    "error": err,
                }
                _append_row_csv(failed_csv, fail_rec)

                if not continue_on_error:
                    raise RuntimeError(err)

    # -------------------------
    # 3) 跑完后生成最终 Excel
    # -------------------------
    if os.path.exists(progress_csv):
        df_all = pd.read_csv(progress_csv)
        df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
        summary = _build_summary(df_ok)

        if os.path.exists(pareto_csv):
            df_pareto = pd.read_csv(pareto_csv)
        else:
            df_pareto = pd.DataFrame()

        with pd.ExcelWriter(final_xlsx, engine="openpyxl") as writer:
            df_all.to_excel(writer, sheet_name="runs", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
            df_pareto.to_excel(writer, sheet_name="pareto_archive", index=False)

        print(f"\n[DONE] Progress CSV : {progress_csv}")
        if update_summary_each_run:
            print(f"[DONE] Live Summary : {summary_live_csv}")
        if os.path.exists(failed_csv):
            print(f"[DONE] Failed Runs  : {failed_csv}")
        if os.path.exists(pareto_csv):
            print(f"[DONE] Pareto CSV   : {pareto_csv}")
        print(f"[DONE] Details Dir   : {details_dir}")
        print(f"[DONE] Final Excel   : {final_xlsx}")

    else:
        print("[DONE] No progress file found; nothing to export.")



if __name__ == "__main__":

    ## ========= 单个文件运行，调试
    instance_name = 'c106_21'
    base_path = r"D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)"
    file_path = os.path.join(base_path, f"{instance_name}.txt")

    res = run_once_ccp(
        instance_path=file_path,
        beta=0.1,
        alpha=0.8,
        run_id=0,
        seed=44410844,
        verbose=1,
        log_every=10,
        K=1
    )

    archive = res["archive"]
    work_history = res["work_history"]
    cand_points = res["cand_points"]

    print(res["runtime_sec"])
    print(res["final_cost"], res["final_ra"])
    print(res["archive_size"])

    plot_pareto_front(
        archive=archive,
        title=f"Pareto Front - {instance_name}"
    )

    # plot_search_evolution(
    #     cand_points=cand_points,
    #     work_history=work_history,
    #     archive=archive,
    #     title=f"Search Evolution - {instance_name}"
    # )




    # # =========================
    # # 你自己的全局参数
    # # =========================
    # BETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # N_RUNS = 10

    # ## ======== 整个文件夹运行
    # path = r'D:\02_Research\DataSet\SPR'

    # # run_experiments_resume_ultimate(
    # #     BETAS,
    # #     N_RUNS,
    # #     instance_source=path,
    # #     results_dir=r"D:\02_Research\Results",
    # # )


    # run_experiments_resume_ultimate_10_worker(
    #     BETAS=BETAS,
    #     N_RUNS=N_RUNS,
    #     instance_source=path,
    #     results_dir=r"D:\02_Research\Results",
    #     base_seed=42,
    #     update_summary_each_run=True,
    #     continue_on_error=True,
    # )







