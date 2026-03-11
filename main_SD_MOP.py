from dataclasses import dataclass
import os
import random
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
from datetime import datetime

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

def pareto_insert_strict(archive, cand, eps=1e-9):
    for s in archive:
        if dominates(s, cand, eps) or same_point(s, cand, eps):
            return archive, False

    new_archive = [s for s in archive if not dominates(cand, s, eps)]
    new_archive.append(cand)
    return new_archive, True

def truncate_archive_simple(archive, max_size=80):
    if len(archive) <= max_size:
        return archive

    by_cost = sorted(archive, key=lambda s: s["cost"])[:max_size // 2]
    by_ra   = sorted(archive, key=lambda s: s["ra"])[:max_size // 2]

    merged = []
    for s in by_cost + by_ra:
        if not any(same_point(s, t) for t in merged):
            merged.append(s)

    if len(merged) < max_size:
        rest = sorted(archive, key=lambda s: (s["cost"], s["ra"]))
        for s in rest:
            if len(merged) >= max_size:
                break
            if not any(same_point(s, t) for t in merged):
                merged.append(s)

    return merged[:max_size]

def pick_work_solution(archive, eps=1e-12):
    if not archive:
        return None

    costs = [s["cost"] for s in archive]
    ras = [s["ra"] for s in archive]
    cmin, cmax = min(costs), max(costs)
    rmin, rmax = min(ras), max(ras)

    def norm(x, lo, hi):
        return 0.0 if abs(hi - lo) < eps else (x - lo) / (hi - lo)

    def score(s):
        c = norm(s["cost"], cmin, cmax)
        r = norm(s["ra"], rmin, rmax)
        return (c * c + r * r) ** 0.5

    return min(archive, key=score)




# =========================
# 你自己的全局参数
# =========================
BETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
N_RUNS = 10


def run_once(
    instance_path: str,
    beta: float,
    run_id: int,
    seed: int,
    return_details: bool = False,
    verbose: int = 1,
    log_every: int = 20,
    max_iter = 2000,
    patience = 30
):
    """
    单次实验：给定 beta + seed，运行完整多目标 VNS + VND，并返回结果 dict。

    目标：
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
        True  -> 额外返回 archive / work_history / final_best / debug_log 等调试对象
    verbose       : int
        0 -> 不打印
        1 -> 每 log_every 轮打印摘要
        2 -> 每轮打印摘要
        3 -> 每个 candidate 都打印
    log_every     : int
        verbose=1 时的打印间隔
    """
    
    # -----------------------------
    # 0. 初始化
    # -----------------------------
    t_stage2 = 0.0
    instance_name = os.path.splitext(os.path.basename(instance_path))[0]
    debug_log = []

    ev_params = {
        "soc_max": 40.0,
        "vehicle_capacity": 650.0,
    }

    problem = load_problem(instance_path, ev_params)
    evaluator = Evaluator(problem, ra_safe=0.1, ra_risk=0.7)

    start_time = time.time()

    # -----------------------------
    # 1. 场景生成
    # -----------------------------
    mu = np.asarray(problem.demand, dtype=float)
    scenarios = generate_normal_scenarios(
        mu=mu,
        beta=beta,
        n_scenarios=100,
        random_state=seed,
    )

    # -----------------------------
    # 2. clustering
    # -----------------------------
    clusters = clustering(
        problem,
        method="kmeans",
        feature="location",
        random_state=1,
    )

    depot_coords = np.asarray(
        [(problem.nodes[i].x, problem.nodes[i].y) for i in problem.depots],
        dtype=float,
    )

    # -----------------------------
    # 3. 初始化：GB + GA
    # -----------------------------
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
            random_state=None,
        )

        gbs_customer_id = [[customer_ids[j] for j in cl] for cl in gbs_info.gbs]


        # params_GB = ACOParams(num_ants=10, num_iter=60)
        # gb_order_result = plan_gb_order(gbs_info.centers, depot_coords, params_GB)

        # params_customer = ACOParams(num_ants=20, num_iter=100, seed=0)
        # routes = plan_internal_gbs_order(
        #     problem,
        #     gb_order_result.order,
        #     gbs_info.centers,
        #     gbs_customer_id,
        #     params=params_customer,
        # )
        # routes_clusters[cid] = routes


        # ===== 规划 gbs 之间的路径（GA）=====
        params_GB = GAParams(
            pop_size=50,
            num_gen=200,
            tournament_k=3,
            crossover_rate=0.9,
            mutation_rate=0.2,
            inversion_rate=0.7,
            elite_size=2,
            seed=0,
        )
        gb_order_result = plan_gb_order_ga(gbs_info.centers, depot_coords, params_GB)

        # ===== 规划 gb 内客户的路径（GA）=====
        params_customer = GAParams(
            pop_size=80,
            num_gen=300,
            tournament_k=3,
            crossover_rate=0.9,
            mutation_rate=0.3,
            inversion_rate=0.7,
            elite_size=2,
            seed=0,
        )
        routes = plan_internal_gbs_order_ga(
            problem,
            gb_order_result.order,
            gbs_info.centers,
            gbs_customer_id,
            params=params_customer,
        )
        routes_clusters[cid] = routes

    # -----------------------------
    # 4. 初始解 SAA 评估
    # -----------------------------
    t0 = time.time()
    init_eval, routes_clusters = evaluator.evaluate_init_saa(routes_clusters, scenarios)
    t_stage2 += time.time() - t0

    init_sol = pack_solution(routes_clusters, init_eval)

    archive = []
    archive, _ = pareto_insert_strict(archive, init_sol, eps=1e-9)
    archive = truncate_archive_simple(archive, max_size=80)

    work = pick_work_solution(archive)
    if work is None:
        raise RuntimeError("Archive is empty after initialization.")

    if verbose >= 1:
        print(
            f"[Init] instance={instance_name} beta={beta:.3f} seed={seed} "
            f"cost={work['cost']:.6f} ra={work['ra']:.6f} archive={len(archive)}"
        )

    work_history = [(0, float(work["cost"]), float(work["ra"]))]
    cand_points = []

    # -----------------------------
    # 5. Multi-objective VNS + VND
    # -----------------------------
    rng = np.random.default_rng(seed)

    nb = Neighborhood(
        problem=problem,
        removal_ratio=0.30,
        alpha_boundary=0.50,
        min_gb_len=2,
        rng=rng,
        evaluator=None,
    )

    shaker = Shaker(nb, NS)

    vnd = VNDRefiner(
        nb,
        evaluator=lambda routes: evaluator.evaluate_saa(routes, scenarios),   # full
        eval_proxy=lambda r: evaluator.evaluate_saa(r, scenarios[:5]),       # proxy
        dominates_fn=dominates,
        rng=rng,
        eps=1e-9,
    )

    eps = 1e-9
    
    no_improve = 0
    k_max = len(NS)
    A_MAX = 80

    for it in range(1, max_iter + 1):
        archive_changed = False
        base = copy.deepcopy(work)
        iter_t0 = time.time()

        if verbose >= 2:
            print(
                f"[Iter {it:04d} | start] "
                f"work=(cost={work['cost']:.6f}, ra={work['ra']:.6f}) "
                f"archive={len(archive)}"
            )

        k = 1
        while k <= k_max:
            base_routes = copy.deepcopy(base["routes_by_clusters"])

            # 1) shaking
            cand_routes = shaker.shake(base_routes, k=k, cid=None)

            # 2) VND refine
            cand_routes, cand_cend, cand_info = vnd.refine(
                cand_routes,
                NL=NL,
                cid=None,
                tries_per_op=2,
                max_steps=30,
            )

            t_stage2 += float(cand_info.get("t_stage2", 0.0))

            # 3) pack candidate
            cand = pack_solution(cand_routes, cand_cend)
            cand_points.append((float(cand["cost"]), float(cand["ra"])))

            # 4) update archive
            old_points = {(float(s["cost"]), float(s["ra"])) for s in archive}

            archive, _ = pareto_insert_strict(archive, cand, eps=eps)
            archive = truncate_archive_simple(archive, max_size=A_MAX)

            new_points = {(float(s["cost"]), float(s["ra"])) for s in archive}
            if new_points != old_points:
                archive_changed = True

            # 5) acceptance
            accepted = dominates(cand, base, eps)

            if verbose >= 3:
                print(
                    f"    [Iter {it:04d} | k={k}] "
                    f"cand=(cost={cand['cost']:.6f}, ra={cand['ra']:.6f}) "
                    f"base=(cost={base['cost']:.6f}, ra={base['ra']:.6f}) "
                    f"accept={accepted} archive={len(archive)}"
                )

            if accepted:
                base = cand
                k = 1
            else:
                k += 1

        # 一轮结束后，从 archive 里选折中工作解
        work = pick_work_solution(archive)
        if work is None:
            raise RuntimeError("Archive became empty during search.")

        work_history.append((it, float(work["cost"]), float(work["ra"])))

        # 基于 archive 是否变化的早停
        if archive_changed:
            no_improve = 0
        else:
            no_improve += 1

        iter_time = time.time() - iter_t0

        best_cost_sol = min(archive, key=lambda s: s["cost"])
        best_ra_sol = min(archive, key=lambda s: s["ra"])

        debug_log.append({
            "iter": int(it),
            "work_cost": float(work["cost"]),
            "work_ra": float(work["ra"]),
            "base_cost": float(base["cost"]),
            "base_ra": float(base["ra"]),
            "best_cost_in_archive": float(best_cost_sol["cost"]),
            "ra_of_best_cost": float(best_cost_sol["ra"]),
            "best_ra_in_archive": float(best_ra_sol["ra"]),
            "cost_of_best_ra": float(best_ra_sol["cost"]),
            "archive_size": int(len(archive)),
            "archive_changed": bool(archive_changed),
            "no_improve": int(no_improve),
            "iter_time": float(iter_time),
            "cum_t_stage2": float(t_stage2),
            "n_candidates_so_far": int(len(cand_points)),
        })

        if verbose >= 1 and (verbose >= 2 or it % log_every == 0):
            print(
                f"[Iter {it:04d} | end] "
                f"work=(cost={work['cost']:.6f}, ra={work['ra']:.6f}) "
                f"best_cost={best_cost_sol['cost']:.6f} "
                f"best_ra={best_ra_sol['ra']:.6f} "
                f"archive={len(archive)} "
                f"changed={archive_changed} "
                f"no_improve={no_improve} "
                f"cand={len(cand_points)} "
                f"time={iter_time:.2f}s"
            )

        if no_improve >= patience:
            if verbose >= 1:
                print(f"[Stop] no improvement for {patience} outer iterations.")
            break

    # -----------------------------
    # 6. 收尾统计
    # -----------------------------
    runtime = time.time() - start_time
    t_stage1 = runtime - t_stage2

    final_best = pick_work_solution(archive)
    if final_best is None:
        raise RuntimeError("Archive is empty at the end of search.")

    min_cost_sol = min(archive, key=lambda s: s["cost"])
    min_ra_sol = min(archive, key=lambda s: s["ra"])

    if verbose >= 1:
        print(
            f"[Final] final_best=(cost={final_best['cost']:.6f}, ra={final_best['ra']:.6f}) "
            f"best_cost={min_cost_sol['cost']:.6f} "
            f"best_ra={min_ra_sol['ra']:.6f} "
            f"archive={len(archive)} runtime={runtime:.2f}s"
        )

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "instance": instance_name,
        "beta": float(beta),
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
        "n_candidates": int(len(cand_points)),
        "n_iterations": int(len(work_history) - 1),

        # 这两个对应 final_best（折中解）
        "final_best_stage1_cost": float(final_best["cend"].get("stage1_cost", np.nan)),
        "final_best_stage2_cost": float(final_best["cend"].get("stage2_cost", np.nan)),

        "runtime_sec": float(runtime),
        "saa_S": int(scenarios.shape[0]),
        "proxy_S": 5,
        "t_stage1": float(t_stage1),
        "t_stage2": float(t_stage2),
    }

    if return_details:
        result.update({
            "archive": archive,
            "work_history": work_history,
            "final_best": final_best,
            "min_cost_sol": min_cost_sol,
            "min_ra_sol": min_ra_sol,
            "problem": problem,
            "scenarios": scenarios,
            "cand_points": cand_points,
            "debug_log": debug_log,
            "clusters": clusters,
        })

    return result



def improves_any(cand, cur, eps):
    # return (
    #     cand["cost"] < cur["cost"] - eps
    #     or cand["ra"]   < cur["ra"]   - eps
    # )
    return cand["cost"] < cur["cost"] - eps

def run_once1(
    instance_path: str,
    beta: float,
    run_id: int,
    seed: int,
    return_details: bool = False,
    verbose: int = 1,
    log_every: int = 20,
    K: int = 5,
    A_MAX: int = 80,
    max_iter: int = 2000,
    patience: int = 100,
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

    # =============================
    # local helpers
    # =============================
    def pareto_set_insert(pareto_set, cand, eps=1e-9):
        """
        在局部集合中维护非支配解。
        与全局 archive 不同，这里只是当前 k 下的局部候选集合。
        """
        dominated = False
        new_set = []
        for s in pareto_set:
            if dominates(s, cand, eps):
                dominated = True
                break
            if not dominates(cand, s, eps):
                new_set.append(s)
        if dominated:
            return pareto_set
        new_set.append(cand)
        return new_set

    def choose_from_improvers(improvers):
        """
        从能支配 base 的 improvers 中选一个。
        这里沿用你 GBACOPlanner 里的简单 tie-break：选 cost 最小。
        你也可以改成 ideal-point distance。
        """
        return min(improvers, key=lambda s: s["cost"])

    # =============================
    # 0. 初始化
    # =============================
    t_stage2 = 0.0
    instance_name = os.path.splitext(os.path.basename(instance_path))[0]
    debug_log = []

    ev_params = {
        "soc_max": 40.0,
        "vehicle_capacity": 650.0,
    }

    problem = load_problem(instance_path, ev_params)
    evaluator = Evaluator(problem, ra_safe=0.1, ra_risk=0.7)

    start_time = time.time()

    # =============================
    # 1. 场景生成
    # =============================
    mu = np.asarray(problem.demand, dtype=float)
    scenarios = generate_normal_scenarios(
        mu=mu,
        beta=beta,
        n_scenarios=100,
        random_state=seed,
    )

    # =============================
    # 2. clustering
    # =============================
    clusters = clustering(
        problem,
        method="kmeans",
        feature="location",
        random_state=1,
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
            split_k=0.6,
            merge=True,
            random_state=None,
        )

        gbs_customer_id = [[customer_ids[j] for j in cl] for cl in gbs_info.gbs]

        # ===== 规划 GB 之间路径（GA）=====
        params_gb = GAParams(
            pop_size=50,
            num_gen=200,
            tournament_k=3,
            crossover_rate=0.9,
            mutation_rate=0.2,
            inversion_rate=0.7,
            elite_size=2,
            seed=0,
        )
        gb_order_result = plan_gb_order_ga(
            gbs_info.centers,
            depot_coords,
            params_gb,
        )

        # ===== 规划 GB 内客户路径（GA）=====
        params_customer = GAParams(
            pop_size=80,
            num_gen=300,
            tournament_k=3,
            crossover_rate=0.9,
            mutation_rate=0.3,
            inversion_rate=0.7,
            elite_size=2,
            seed=0,
        )
        routes = plan_internal_gbs_order_ga(
            problem,
            gb_order_result.order,
            gbs_info.centers,
            gbs_customer_id,
            params=params_customer,
        )

        routes_clusters[cid] = routes

    # =============================
    # 4. 初始解 SAA 评估
    # =============================
    t0 = time.time()
    init_eval, routes_clusters = evaluator.evaluate_init_saa(routes_clusters, scenarios)
    t_stage2 += time.time() - t0

    init_sol = pack_solution(routes_clusters, init_eval)

    archive = []
    archive, _ = pareto_insert_strict(archive, init_sol, eps=1e-9)
    archive = truncate_archive_simple(archive, max_size=A_MAX)

    work = pick_work_solution(archive)
    if work is None:
        raise RuntimeError("Archive is empty after initialization.")

    if verbose >= 1:
        print(
            f"[Init] instance={instance_name} beta={beta:.3f} seed={seed} "
            f"cost={work['cost']:.6f} ra={work['ra']:.6f} archive={len(archive)}"
        )

    work_history = [(0, float(work["cost"]), float(work["ra"]))]
    cand_points = [(float(init_sol["cost"]), float(init_sol["ra"]))]

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
        evaluator=None,
    )

    shaker = Shaker(nb, NS)

    vnd = VNDRefiner(
        nb,
        evaluator=lambda routes: evaluator.evaluate_saa(routes, scenarios),   # full eval
        eval_proxy=lambda r: evaluator.evaluate_saa(r, scenarios[:5]),       # proxy eval
        dominates_fn=improves_any,
        rng=rng,
        eps=1e-9,
    )

    eps = 1e-9
    no_improve = 0
    k_max = len(NS)

    for it in range(1, max_iter + 1):
        archive_changed = False
        base = copy.deepcopy(work)
        iter_t0 = time.time()

        if verbose >= 2:
            print(
                f"[Iter {it:04d} | start] "
                f"work=(cost={work['cost']:.6f}, ra={work['ra']:.6f}) "
                f"archive={len(archive)}"
            )

        k = 1
        while k <= k_max:
            best_local_set = []

            # ---------------------------------
            # 在当前 k 下，采样 K 个候选
            # ---------------------------------
            for kk in range(K):
                base_routes = copy.deepcopy(base["routes_by_clusters"])

                # 1) shaking
                cand_routes = shaker.shake(base_routes, k=k, cid=None)

                # 2) VND refine
                cand_routes, cand_cend, cand_info = vnd.refine(
                    cand_routes,
                    NL=NL,
                    cid=None,
                    tries_per_op=2,
                    max_steps=30,
                )

                t_stage2 += float(cand_info.get("t_stage2", 0.0))

                # 3) pack candidate
                cand = pack_solution(cand_routes, cand_cend)
                cand_points.append((float(cand["cost"]), float(cand["ra"])))

                # 4) 更新全局 archive
                old_points = {(float(s["cost"]), float(s["ra"])) for s in archive}

                archive, _ = pareto_insert_strict(archive, cand, eps=eps)
                archive = truncate_archive_simple(archive, max_size=A_MAX)

                new_points = {(float(s["cost"]), float(s["ra"])) for s in archive}
                if new_points != old_points:
                    archive_changed = True

                # 5) 更新当前 k 下的局部非支配集
                best_local_set = pareto_set_insert(best_local_set, cand, eps=eps)

                if verbose >= 3:
                    accepted_now = dominates(cand, base, eps)
                    print(
                        f"    [Iter {it:04d} | k={k} | sample={kk+1}/{K}] "
                        f"cand=(cost={cand['cost']:.6f}, ra={cand['ra']:.6f}) "
                        f"base=(cost={base['cost']:.6f}, ra={base['ra']:.6f}) "
                        f"direct_dom={accepted_now} "
                        f"local_nd={len(best_local_set)} archive={len(archive)}"
                    )

            # ---------------------------------
            # 在 best_local_set 中判断是否有点支配 base
            # ---------------------------------
            improvers = [s for s in best_local_set if dominates(s, base, eps)]

            if improvers:
                chosen = choose_from_improvers(improvers)
                base = chosen
                k = 1

                if verbose >= 3:
                    print(
                        f"        -> accept from local set: "
                        f"(cost={base['cost']:.6f}, ra={base['ra']:.6f})"
                    )
            else:
                k += 1

        # ---------------------------------
        # 一轮结束：从 archive 挑下一个 work
        # ---------------------------------
        work = pick_work_solution(archive)
        if work is None:
            raise RuntimeError("Archive became empty during search.")

        work_history.append((it, float(work["cost"]), float(work["ra"])))

        # early stop: 基于 archive 是否变化
        if archive_changed:
            no_improve = 0
        else:
            no_improve += 1

        iter_time = time.time() - iter_t0

        best_cost_sol = min(archive, key=lambda s: s["cost"])
        best_ra_sol = min(archive, key=lambda s: s["ra"])

        debug_log.append({
            "iter": int(it),
            "work_cost": float(work["cost"]),
            "work_ra": float(work["ra"]),
            "base_cost": float(base["cost"]),
            "base_ra": float(base["ra"]),
            "best_cost_in_archive": float(best_cost_sol["cost"]),
            "ra_of_best_cost": float(best_cost_sol["ra"]),
            "best_ra_in_archive": float(best_ra_sol["ra"]),
            "cost_of_best_ra": float(best_ra_sol["cost"]),
            "archive_size": int(len(archive)),
            "archive_changed": bool(archive_changed),
            "no_improve": int(no_improve),
            "iter_time": float(iter_time),
            "cum_t_stage2": float(t_stage2),
            "n_candidates_so_far": int(len(cand_points)),
        })

        if verbose >= 1 and (verbose >= 2 or it % log_every == 0):
            print(
                f"[Iter {it:04d} | end] "
                f"work=(cost={work['cost']:.6f}, ra={work['ra']:.6f}) "
                f"best_cost={best_cost_sol['cost']:.6f} "
                f"best_ra={best_ra_sol['ra']:.6f} "
                f"archive={len(archive)} "
                f"changed={archive_changed} "
                f"no_improve={no_improve} "
                f"cand={len(cand_points)} "
                f"time={iter_time:.2f}s"
            )

        if no_improve >= patience:
            if verbose >= 1:
                print(f"[Stop] no improvement for {patience} outer iterations.")
            break

    # =============================
    # 6. 收尾统计
    # =============================
    runtime = time.time() - start_time
    t_stage1 = runtime - t_stage2

    final_best = pick_work_solution(archive)
    if final_best is None:
        raise RuntimeError("Archive is empty at the end of search.")

    min_cost_sol = min(archive, key=lambda s: s["cost"])
    min_ra_sol = min(archive, key=lambda s: s["ra"])

    if verbose >= 1:
        print(
            f"[Final] final_best=(cost={final_best['cost']:.6f}, ra={final_best['ra']:.6f}) "
            f"best_cost={min_cost_sol['cost']:.6f} "
            f"best_ra={min_ra_sol['ra']:.6f} "
            f"archive={len(archive)} runtime={runtime:.2f}s"
        )

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "instance": instance_name,
        "beta": float(beta),
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
        "n_candidates": int(len(cand_points)),
        "n_iterations": int(len(work_history) - 1),

        # 对应 final_best（折中解）
        "final_best_stage1_cost": float(final_best["cend"].get("stage1_cost", np.nan)),
        "final_best_stage2_cost": float(final_best["cend"].get("stage2_cost", np.nan)),

        "runtime_sec": float(runtime),
        "saa_S": int(scenarios.shape[0]),
        "proxy_S": 5,
        "t_stage1": float(t_stage1),
        "t_stage2": float(t_stage2),
    }

    if return_details:
        result.update({
            "archive": archive,
            "work_history": work_history,
            "final_best": final_best,
            "min_cost_sol": min_cost_sol,
            "min_ra_sol": min_ra_sol,
            "problem": problem,
            "scenarios": scenarios,
            "cand_points": cand_points,
            "debug_log": debug_log,
            "clusters": clusters,
        })

    return result


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
    # beta round 防止 0.3000000004 浮点坑
    return (str(instance), round(float(beta), 6), int(run_id))


# =========================
# 工具函数：安全追加一行 CSV
# =========================
def _append_row_csv(path: str, record: dict):
    df_row = pd.DataFrame([record])
    write_header = not os.path.exists(path)
    df_row.to_csv(path, mode="a", header=write_header, index=False)


# =========================
# 工具函数：从 progress 构建 summary
# =========================
def _build_summary(df: pd.DataFrame):
    # 你可以按需把更多指标加入 summary
    agg_dict = {
        "best_cost": ["count", "mean", "std", "min"],
        "runtime_sec": ["mean", "std"],
    }

    # 如果有两阶段时间，就加进去
    if "t_stage1" in df.columns:
        agg_dict["t_stage1"] = ["mean", "std"]
    if "t_stage2" in df.columns:
        agg_dict["t_stage2"] = ["mean", "std"]

    summary = (
        df.groupby(["instance", "beta"])
          .agg(agg_dict)
    )

    # flatten columns
    summary.columns = ["_".join(col).strip() for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()

    # 可选：stage2 占比
    if "t_stage2_mean" in summary.columns and "runtime_sec_mean" in summary.columns:
        summary["stage2_ratio_mean"] = summary["t_stage2_mean"] / summary["runtime_sec_mean"]

    return summary


# =========================
# 终极稳定版：断点续跑 + 实时落盘 + 实时均值
# =========================
def run_experiments_resume_ultimate(
    instance_source: str,
    results_dir: Optional[str] = None,
    base_seed: int = 42,
    update_summary_each_run: bool = True,   # ✅ 是否每次 run 结束都更新 summary_live.csv
    continue_on_error: bool = True,         # ✅ 出错是否继续跑下一个任务
):
    """
    终极稳定版：
    - progress_runs.csv：每次 run 结束立即追加（核心数据源）
    - summary_live.csv：可选，每次 run 更新均值（你可以随时打开看当前均值）
    - failed_runs.csv：记录失败的任务，含错误信息
    - final_results.xlsx：最终导出 runs + summary
    - 重启自动跳过已完成项
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
    final_xlsx = os.path.join(results_dir, "final_results.xlsx")

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

                # seed 规则（稳定、可复现、避免冲突）
                seed = base_seed + 1000 * int(round(beta * 10)) + (run_id - 1)

                tasks.append((instance_path, instance_name, beta, run_id, seed))

    print(f"[TASKS] total={total_tasks} | skipped(done)={skipped} | remaining={len(tasks)}")

    # 如果没有任务了，也可以直接生成 final
    if len(tasks) == 0 and os.path.exists(progress_csv):
        df_all = pd.read_csv(progress_csv)
        summary = _build_summary(df_all)
        with pd.ExcelWriter(final_xlsx, engine="openpyxl") as writer:
            df_all.to_excel(writer, sheet_name="runs", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
        print(f"[DONE] Nothing to run. Final Excel refreshed: {final_xlsx}")
        return

    # -------------------------
    # 2) 执行任务：每次 run 完立即写 CSV（防丢）
    # -------------------------
    for idx, (instance_path, instance_name, beta, run_id, seed) in enumerate(tasks, 1):
        print(f"\n=== [{idx}/{len(tasks)}] instance={instance_name} beta={beta:.1f} run={run_id}/{N_RUNS} seed={seed} ===")

        try:
            rec = run_once(
                instance_path=instance_path,
                beta=beta,
                run_id=run_id,
                seed=seed
            )

            # ✅ 关键列强制补齐（断点续跑依赖这些列）
            rec["instance"] = rec.get("instance", instance_name)
            rec["beta"] = float(rec.get("beta", beta))
            rec["run_id"] = int(rec.get("run_id", run_id))
            rec["seed"] = int(rec.get("seed", seed))
            rec["status"] = "completed"

            # ✅ 追加保存（核心）
            _append_row_csv(progress_csv, rec)

            # 更新 done
            done.add(_make_key(rec["instance"], rec["beta"], rec["run_id"]))

            # ✅ 可选：实时更新 summary（方便你随时看均值）
            if update_summary_each_run:
                df_all = pd.read_csv(progress_csv)
                if "status" in df_all.columns:
                    df_ok = df_all[df_all["status"] == "completed"].copy()
                else:
                    df_ok = df_all.copy()
                summary = _build_summary(df_ok)
                summary.to_csv(summary_live_csv, index=False)

        except Exception as e:
            err = repr(e)
            print(f"[ERROR] instance={instance_name} beta={beta} run={run_id} failed: {err}")

            # 失败也落盘（方便你之后重跑/统计失败率）
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

            # continue 跑下一个
            continue

    # -------------------------
    # 3) 跑完后生成最终 Excel（runs + summary）
    # -------------------------
    if os.path.exists(progress_csv):
        df_all = pd.read_csv(progress_csv)
        # 只用 completed 做统计
        if "status" in df_all.columns:
            df_ok = df_all[df_all["status"] == "completed"].copy()
        else:
            df_ok = df_all.copy()

        summary = _build_summary(df_ok)

        with pd.ExcelWriter(final_xlsx, engine="openpyxl") as writer:
            df_all.to_excel(writer, sheet_name="runs", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)

        print(f"\n[DONE] Progress CSV : {progress_csv}")
        if update_summary_each_run:
            print(f"[DONE] Live Summary : {summary_live_csv}")
        if os.path.exists(failed_csv):
            print(f"[DONE] Failed Runs  : {failed_csv}")
        print(f"[DONE] Final Excel   : {final_xlsx}")
    else:
        print("[DONE] No progress file found; nothing to export.")


# =========================
# 你只需要调用：
# run_experiments_resume_ultimate("path/to/instance_or_folder")
# =========================



if __name__ == "__main__":

    ## ========= 单个文件运行，调试
    instance_name = 'c103_21'
    base_path = r"D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)"
    file_path = os.path.join(base_path, f"{instance_name}.txt")

    res = run_once(
        instance_path=file_path,
        beta=0,
        run_id=0,
        seed=0,
        return_details=True,
        verbose=1,
        log_every=10,
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

    plot_search_evolution(
        cand_points=cand_points,
        work_history=work_history,
        archive=archive,
        title=f"Search Evolution - {instance_name}"
    )



    # ## ======== 整个文件夹运行
    # path = r'D:\02_Research\DataSet\SPR'

    # run_experiments_resume_ultimate(
    #     instance_source=path,
    #     results_dir=r"D:\02_Research\Results"
    # )







