from dataclasses import dataclass
import os
import random
import sys
import time
from typing import Optional

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
    """
    把一个解打包成 archive / Pareto 比较用的统一结构。
    现在除了总 cost / 总 ra，也显式保存两阶段 cost / ra。
    """
    return {
        "routes_by_clusters": copy.deepcopy(routes_by_clusters),

        # 总目标
        "cost": float(cend["total_cost"]),
        "ra": float(cend.get("total_ra", 0.0)),

        # 两阶段 cost
        "stage1_cost": float(cend.get("stage1_cost", np.nan)),
        "stage2_cost": float(cend.get("stage2_cost", np.nan)),

        # 两阶段 RA
        "stage1_ra": float(cend.get("stage1_ra", np.nan)),
        "stage2_ra": float(cend.get("stage2_ra", np.nan)),

        # 保留完整评估结果
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







def improves_any(cand, cur, eps):
    # return (
    #     cand["cost"] < cur["cost"] - eps
    #     or cand["ra"]   < cur["ra"]   - eps
    # )
    return cand["cost"] < cur["cost"] - eps


def run_once_spr(
    instance_path: str,
    beta: float,
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
        random_state=seed,
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
            random_state=seed + cid,
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
            seed=seed + 1000 + cid,
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
            seed=seed + 2000 + cid,
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

    # SAA_WORKERS = 10
    # init_eval, routes_clusters = evaluator.evaluate_init_saa_n_workers(routes_clusters, scenarios, n_workers=SAA_WORKERS)
    
    t_stage2 += time.time() - t0

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
    
    proxy_S = min(5, len(scenarios))
    sample = rng.choice(len(scenarios), size=proxy_S, replace=False)
    proxy_scenarios = scenarios[sample]
    vnd = VNDRefiner(
        nb,
        evaluator=lambda routes: evaluator.evaluate_saa(routes, scenarios),   # full eval
        eval_proxy=lambda r: evaluator.evaluate_saa(r, proxy_scenarios),      # proxy eval
        dominates_fn=dominates,
        rng=rng,
        eps=eps,
    )

    

    # vnd = VNDRefiner(
    #     nb,
    #     evaluator=lambda routes: evaluator.evaluate_saa_n_workers(routes, scenarios, n_workers=SAA_WORKERS),
    #     eval_proxy=lambda r: evaluator.evaluate_saa_n_workers(r, scenarios[:5], n_workers=min(SAA_WORKERS, 5)),
    #     dominates_fn=dominates,
    #     rng=rng,
    #     eps=eps,
    # )

    
    no_improve = 0
    k_max = len(NS)

    for it in range(1, max_iter + 1):
        archive_changed = False
        base = copy.deepcopy(work)
        iter_t0 = time.time()

        # if verbose >= 2:
        #     print(
        #         f"[Iter {it:04d} | start] "
        #         f"work=(cost={work['cost']:.6f}, ra={work['ra']:.6f}) "
        #         f"archive={len(archive)}"
        #     )

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
                archive = truncate_by_crowding(archive, max_size=A_MAX)

                new_points = {(float(s["cost"]), float(s["ra"])) for s in archive}
                if new_points != old_points:
                    archive_changed = True

                # 6) 更新 Nk 内 Pareto 集
                best_local_set = pareto_set_insert(best_local_set, cand, eps=eps)

                # if verbose >= 3:
                #     accepted_now = dominates(cand, base, eps)
                #     print(
                #         f"    [Iter {it:04d} | k={k} | sample={kk+1}/{K}] "
                #         f"cand=(cost={cand['cost']:.6f}, ra={cand['ra']:.6f}) "
                #         f"base=(cost={base['cost']:.6f}, ra={base['ra']:.6f}) "
                #         f"direct_dom={accepted_now} "
                #         f"local_nd={len(best_local_set)} archive={len(archive)}"
                #     )

            # ---------------------------------
            # 在 best_local_set 中判断是否有点支配 base
            # ---------------------------------
            improvers = [s for s in best_local_set if dominates(s, base, eps)]

            if improvers:
                # tie-break：在能改进 base 的点里，选一个（这里先选 cost 最小；也可换成理想点距离）
                # base = min(improvers, key=lambda s: s["cost"])  # tie-break
                base = min(improvers, key=lambda s: (s["cost"], s["ra"]))
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
        work = pick_work_grid(archive)
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

        # if verbose >= 1 and (verbose >= 2 or it % log_every == 0):
        #     print(
        #         f"[Iter {it:04d} | end] "
        #         f"work=(cost={work['cost']:.6f}, ra={work['ra']:.6f}) "
        #         f"best_cost={best_cost_sol['cost']:.6f} "
        #         f"best_ra={best_ra_sol['ra']:.6f} "
        #         f"archive={len(archive)} "
        #         f"changed={archive_changed} "
        #         f"no_improve={no_improve} "
        #         f"cand={len(cand_points)} "
        #         f"time={iter_time:.2f}s"
        #     )

        

    # =============================
    # 6. 收尾统计
    # =============================
    runtime = time.time() - start_time
    t_stage1 = runtime - t_stage2

    final_best = pick_work_grid(archive)
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










if __name__ == "__main__":

    ## ========= 单个文件运行，调试
    instance_name = 'rc103_21'
    base_path = r"D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)"
    file_path = os.path.join(base_path, f"{instance_name}.txt")

    res = run_once_spr(
        instance_path=file_path,
        beta=0.1,
        run_id=0,
        seed=0,
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







