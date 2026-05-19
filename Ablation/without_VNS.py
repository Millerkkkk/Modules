from collections import OrderedDict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

MODULES_ROOT = Path(__file__).resolve().parents[1]   # .../Project_python/Modules
if str(MODULES_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULES_ROOT))

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

from concurrent.futures import ProcessPoolExecutor, as_completed

from stochastic_demand import fast_forward_selection, generate_normal_scenarios





from params import ACOParams, GAParams, ev_params
import copy

from GbPlanning_GA import plan_gb_order_ga
from CustomerPlanning_GA import plan_internal_gbs_order_ga

from GbPlanning import plan_gb_order
from CustomerPlanning import plan_internal_gbs_order

from EVRP.loader import load_problem      # 按你实际路径改
from Clustering.api import clustering               # 你重构后的统一入口
from Clustering.plot import plot_clusters
from GB.api import gb_clustering

from evaluate import Evaluator
from plot import plot_clusters_gbs, plot_clusters_gbs_with_gb_routes, plot_routes, plot_pareto_front, plot_search_evolution
from VNS.neighborhood import Neighborhood
from VNS.shaking import Shaker
from VNS.vnd import VNDRefiner




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
           

NL_ra = [
    "relocate_node_intra_gb",
    "reverse_node_intra_gb",
    "relocate_node_inter_gb",
    "swap_node_inter_gb",
    "relocate_node_inter_route",
]



# ---------- helpers ----------
def pack_solution(routes_by_clusters, cend):
    """
    把一个解打包成 archive / Pareto 比较用的统一结构。
    除了总 cost / 总 ra，也显式保存 stats，方便 run_once_spr / 存档 / 分析直接访问。
    """
    return {
        "routes_by_clusters": copy.deepcopy(routes_by_clusters),

        # 总目标
        "cost": float(cend["total_cost"]),
        "ra": float(cend.get("total_ra", 0.0)),

        # 统计（scenario 聚合后的总 stats）
        "stats": copy.deepcopy(cend.get("stats", {})),

        # 保留完整评估结果
        "cend": copy.deepcopy(cend),
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

def pareto_insert(pareto_set, cand, eps=1e-9):
    for s in pareto_set:
        if dominates(s, cand, eps) or same_point(s, cand, eps):
            return pareto_set, False

    new_archive = [s for s in pareto_set if not dominates(cand, s, eps)]
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




def pareto_update(E, E_new, eps=1e-6, max_size=None):
    merged = list(E)
    for s in E_new:
        merged = pareto_insert(merged, s, eps=eps)

    if max_size is not None:
        merged = truncate_by_crowding(merged, max_size=max_size)
    return merged

def mo_improvement(E, E_new, eps=1e-6):
    old_points = {(round(s["cost"], 8), round(s["ra"], 8)) for s in E}
    updated = pareto_update(E, E_new, eps=eps, max_size=None)
    new_points = {(round(s["cost"], 8), round(s["ra"], 8)) for s in updated}
    return new_points != old_points

def mo_shake(archive, shaker, evaluator, k, cid=None):
    shaken_set = []
    seen = set()

    for sol in archive:
        base_routes = copy.deepcopy(sol["routes_by_clusters"])
        cand_routes = shaker.shake(base_routes, k=k, cid=cid)

        sig = freeze_routes(cand_routes)
        if sig in seen:
            continue
        seen.add(sig)

        cand_cend = evaluator(cand_routes)
        cand = pack_solution(cand_routes, cand_cend)

        shaken_set.append(cand)

    return shaken_set

def mo_vnd(shaken_set, vnd, obj, NL, cid=None, 
           eps=1e-6, max_size=None):
    """
    shaken_set: shake 后的解集合
    vnd: 单解 VND 对象
    obj: 当前优化目标, "cost" 或 "ra"
    返回:
        vnd_set: MOVND 后的 Pareto 集
        info_out: 统计信息
    """
    vnd_set = []

    for sol in shaken_set:
        final_routes, final_cend, info = vnd.refine_ffs(
            sol,
            obj=obj,
            NL=NL,
            cid=cid,
        )

        # 中间访问点也尝试加入 Pareto 集
        seen_sig = set()
        for cand in info.get("visited", []):
            sig = freeze_routes(cand["routes_by_clusters"])
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            vnd_set, _ = pareto_insert(vnd_set, cand, eps=eps)

        # 最终点加入 Pareto 集
        final_sol = pack_solution(final_routes, final_cend)
        vnd_set, _ = pareto_insert(vnd_set, final_sol, eps=eps)

    if max_size is not None:
        vnd_set = truncate_by_crowding(vnd_set, max_size=max_size)

    return vnd_set


def freeze_routes(routes_by_clusters):
    """
    把 routes_by_clusters 转成可哈希对象，作为 cache key。
    你要按你自己的 routes 结构微调。
    """
    frozen = []
    for cid in sorted(routes_by_clusters.keys()):
        cluster_routes = routes_by_clusters[cid]

        # 假设 cluster_routes 是 list[route]
        # route 里是客户 id 列表，比如 [0, 3, 5, 0]
        frozen_cluster = tuple(tuple(route) for route in cluster_routes)
        frozen.append((cid, frozen_cluster))

    return tuple(frozen)



class EvalCache:
    def __init__(self, evaluator_fn, max_size=50000):
        self.evaluator_fn = evaluator_fn
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hit = 0
        self.miss = 0

    def __call__(self, routes):
        key = freeze_routes(routes)

        if key in self.cache:
            self.hit += 1
            # 命中后移动到末尾，表示最近使用
            self.cache.move_to_end(key)
            return copy.deepcopy(self.cache[key])

        self.miss += 1
        val = self.evaluator_fn(routes)

        # 存入前先深拷贝，避免外部对象后续被修改污染缓存
        self.cache[key] = copy.deepcopy(val)
        self.cache.move_to_end(key)

        # 超出容量时，删除最久未使用项（最前面）
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

        return copy.deepcopy(val)

    def stats(self):
        total = self.hit + self.miss
        rate = self.hit / total if total > 0 else 0.0
        return {
            "hit": self.hit,
            "miss": self.miss,
            "hit_rate": rate,
            "size": len(self.cache),
            "max_size": self.max_size,
        }

    def clear(self):
        self.cache.clear()
        self.hit = 0
        self.miss = 0




def run_once_spr2(
    instance_path: str,
    beta: float,
    run_id: int,
    seed: int,
    n_scenarios: int = 50,
    n_reduced_scenarios: int = 10,
    verbose: int = 1,
    log_every: int = 20,
    K: int = 1,
    A_MAX: int = 80,
    max_iter: int = 200,
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
        n_scenarios=n_scenarios,
        random_state=seed,
    )

    selected_idx, reduced_scenarios, reduced_probs = fast_forward_selection(
        scenarios=scenarios,
        n_select=n_reduced_scenarios,
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

    # # =============================
    # # 4. 初始解 SAA 评估
    # # =============================
    # t0 = time.time()
    # init_eval, routes_clusters = evaluator.evaluate_init_saa(routes_clusters, scenarios)

    # # SAA_WORKERS = 10
    # # init_eval, routes_clusters = evaluator.evaluate_init_saa_n_workers(routes_clusters, scenarios, n_workers=SAA_WORKERS)
    
    # =============================
    # 4. 初始解 FFS 评估
    # =============================
    t0 = time.time()
    init_eval, routes_clusters = evaluator.evaluate_init_ffs(routes_clusters, reduced_scenarios, reduced_probs)

    t_stage2 += time.time() - t0

    init_sol = pack_solution(routes_clusters, init_eval)

    archive = []
    archive, _ = pareto_insert(archive, init_sol, eps=eps)
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
        # evaluator=lambda routes: evaluator.evaluate_saa(routes, scenarios),   # full eval
        evaluator=lambda routes: evaluator.evaluate_ffs(routes, reduced_scenarios, reduced_probs),   # full eval
        eval_proxy=lambda r: evaluator.evaluate_ffs(r, reduced_scenarios, reduced_probs),      # proxy eval
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

                archive, _ = pareto_insert(archive, cand, eps=eps)
                archive = truncate_by_crowding(archive, max_size=A_MAX)

                new_points = {(float(s["cost"]), float(s["ra"])) for s in archive}
                if new_points != old_points:
                    archive_changed = True

                # 6) 更新 Nk 内 Pareto 集
                best_local_set, _ = pareto_insert(best_local_set, cand, eps=eps)

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





def run_once_spr1(
    instance_path: str,
    beta: float,
    run_id: int,
    seed: int,
    n_scenarios: int = 50,
    n_reduced_scenarios: int = 10,
    verbose: int = 1,
    log_every: int = 20,
    A_MAX: int = 60,
    max_iter: int = 200,
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
    debug_log = []
    start_time = time.time()

    # =============================
    # 0. Problem
    # =============================
    problem = load_problem(instance_path, ev_params)
    instance_name = os.path.splitext(os.path.basename(instance_path))[0]

    # =============================
    # 1. Scenarios
    # =============================
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

    # =============================
    # 2. Evaluator / search objects
    # =============================
    rng = np.random.default_rng(seed)

    evaluator = Evaluator(problem, ra_safe=0.1, ra_risk=0.7)

    base_eval_fn = lambda routes: evaluator.evaluate_ffs(
        routes, reduced_scenarios, reduced_probs
    )
    cached_eval = EvalCache(base_eval_fn, max_size=30000)

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
        evaluator=cached_eval,
        eval_proxy=cached_eval,
        dominates_fn=dominates,
        rng=rng,
        eps=eps,
    )
    
    # =============================
    # 3. Clustering
    # =============================
    depot_coords = np.asarray(
        [(problem.nodes[i].x, problem.nodes[i].y) for i in problem.depots],
        dtype=float,
    )

    clusters = clustering(
        problem,
        method="kmeans",
        feature="location",
        random_state=seed,
    )

    # =============================
    # 4. GB + GA initialization
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

    # # =============================
    # # 4. 初始解 SAA 评估
    # # =============================
    # t0 = time.time()
    # init_eval, routes_clusters = evaluator.evaluate_init_saa(routes_clusters, scenarios)

    # # SAA_WORKERS = 10
    # # init_eval, routes_clusters = evaluator.evaluate_init_saa_n_workers(routes_clusters, scenarios, n_workers=SAA_WORKERS)

    
    # =============================
    # 5. Initial solution by FFS
    # =============================
    init_eval, routes_clusters = evaluator.evaluate_init_ffs(
        routes_clusters, reduced_scenarios, reduced_probs
    )
    init_sol = pack_solution(routes_clusters, init_eval)

    archive = []
    archive, _ = pareto_insert(archive, init_sol, eps=eps)
    if A_MAX is not None:
        archive = truncate_by_crowding(archive, max_size=A_MAX)

    cand_points = [(float(init_sol["cost"]), float(init_sol["ra"]))]


    # =============================
    # 5. Multi-objective VNS + VND
    # =============================
    no_improve = 0
    k_max = len(NS)

    for it in range(1, max_iter + 1):
        archive_changed = False
        iter_t0 = time.time()

        k = 1
        while k <= k_max:
            old_points = {(float(s["cost"]), float(s["ra"])) for s in archive}

            # 1) MO-Shake
            archive_shake = mo_shake(
                archive=archive,
                shaker=shaker,
                evaluator=cached_eval,
                k=k,
                cid=None,
            )

            for sol in archive_shake:
                cand_points.append((float(sol["cost"]), float(sol["ra"])))

            # 2) VND on cost
            archive_vnd_cost = mo_vnd(
                shaken_set=archive_shake,
                vnd=vnd,
                obj="cost",
                NL=NL,
                cid=None,
                eps=eps,
                max_size=A_MAX,
            )

            # 3) VND on ra
            archive_vnd_ra = mo_vnd(
                shaken_set=archive_vnd_cost,
                vnd=vnd,
                obj="ra",
                NL=NL,
                cid=None,
                eps=eps,
                max_size=A_MAX,
            )

            for sol in archive_vnd_ra:
                cand_points.append((float(sol["cost"]), float(sol["ra"])))

            # 4) merge into archive
            archive_new = list(archive)
            for sol in archive_vnd_ra:
                archive_new, _ = pareto_insert(archive_new, sol, eps=eps)

            if A_MAX is not None:
                archive_new = truncate_by_crowding(archive_new, max_size=A_MAX)

            new_points = {(float(s["cost"]), float(s["ra"])) for s in archive_new}

            if new_points != old_points:
                archive = archive_new
                archive_changed = True
                k = 1
            else:
                k += 1

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

        if verbose >= 2 or (verbose >= 1 and (it % log_every) == 0):
            print(
                f"[Iter {it}] archive={len(archive)} "
                f"best_cost={best_cost_sol['cost']:.3f} "
                f"best_ra={best_ra_sol['ra']:.6f} "
                f"changed={archive_changed}"
            )

        debug_log.append({
            "iter": int(it),
            "best_cost_in_archive": float(best_cost_sol["cost"]),
            "ra_of_best_cost": float(best_cost_sol["ra"]),
            "best_ra_in_archive": float(best_ra_sol["ra"]),
            "cost_of_best_ra": float(best_ra_sol["cost"]),
            "archive_size": int(len(archive)),
            "archive_changed": bool(archive_changed),
            "no_improve": int(no_improve),
            "iter_time": float(iter_time),
            "n_candidates_so_far": int(len(cand_points)),
        })

    # =============================
    # 7. Final summary
    # =============================
    runtime = time.time() - start_time

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

        "runtime_sec": float(runtime),
        "saa_S": int(scenarios.shape[0]),
        "n_reduced_scenarios": int(reduced_scenarios.shape[0]),

        "pareto_points": [(float(sol["cost"]), float(sol["ra"])) for sol in archive],
        "archive": archive,
        "final_best": final_best,
        "min_cost_sol": min_cost_sol,
        "min_ra_sol": min_ra_sol,
        "cand_points": cand_points,
        "clusters": clusters,
        "selected_idx": selected_idx,
        "debug_log": debug_log,
        "n_iterations": int(len(debug_log)),
    }

    return result





def append_result_to_summary_csv(result, csv_path):
    """
    把单次 run 的 result 追加写入 summary.csv
    只保留标量字段，避免 archive/debug_log/final_best 这类复杂对象塞进 CSV。
    """
    row = {}

    for k, v in result.items():
        if isinstance(v, (str, int, float, bool, np.integer, np.floating)):
            row[k] = v
        elif v is None:
            row[k] = None

    df_new = pd.DataFrame([row])

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        # 列对齐，避免不同 run 字段数不一致
        all_cols = list(dict.fromkeys(list(df_old.columns) + list(df_new.columns)))
        df_old = df_old.reindex(columns=all_cols)
        df_new = df_new.reindex(columns=all_cols)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

def append_debug_log_to_iter_csv(result, csv_path):
    """
    把当前 run 的 debug_log 追加到总 iter_log.csv
    """
    debug_log = result.get("debug_log", [])
    if not debug_log:
        return

    df_new = pd.DataFrame(debug_log)

    df_new.insert(0, "instance", result.get("instance"))
    df_new.insert(1, "beta", result.get("beta"))
    df_new.insert(2, "run_id", result.get("run_id"))
    df_new.insert(3, "seed", result.get("seed"))

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        all_cols = list(dict.fromkeys(list(df_old.columns) + list(df_new.columns)))
        df_old = df_old.reindex(columns=all_cols)
        df_new = df_new.reindex(columns=all_cols)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

def save_debug_log_csv(result, csv_path):
    """
    把 result["debug_log"] 导出成 iter_log.csv
    并自动补 instance/beta/run_id/seed，方便后续合并多次实验。
    """
    debug_log = result.get("debug_log", [])
    if not debug_log:
        return

    df = pd.DataFrame(debug_log)

    # 补实验标识列，方便多 run 汇总
    df.insert(0, "instance", result.get("instance"))
    df.insert(1, "beta", result.get("beta"))
    df.insert(2, "run_id", result.get("run_id"))
    df.insert(3, "seed", result.get("seed"))

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

def archive_stats(arch):
    """
    返回 archive 的一些基础统计量
    """
    if not arch:
        return {
            "best_cost": np.nan,
            "best_ra": np.nan,
            "worst_cost": np.nan,
            "worst_ra": np.nan,
            "spread_cost": 0.0,
            "spread_ra": 0.0,
        }

    costs = np.asarray([float(s["cost"]) for s in arch], dtype=float)
    ras = np.asarray([float(s["ra"]) for s in arch], dtype=float)

    return {
        "best_cost": float(costs.min()),
        "best_ra": float(ras.min()),
        "worst_cost": float(costs.max()),
        "worst_ra": float(ras.max()),
        "spread_cost": float(costs.max() - costs.min()),
        "spread_ra": float(ras.max() - ras.min()),
    }


def hypervolume_2d_min(arch, ref_point, eps=1e-12):
    """
    二目标最小化问题的 2D Hypervolume
    arch: Pareto archive，目标为 minimize(cost), minimize(ra)
    ref_point: (ref_cost, ref_ra)，必须比 archive 中所有点更差
    """
    if not arch:
        return 0.0

    rc, rr = float(ref_point[0]), float(ref_point[1])

    # 去重 + 只保留落在参考点以内的点
    pts = []
    seen = set()
    for s in arch:
        c = float(s["cost"])
        r = float(s["ra"])
        if c <= rc + eps and r <= rr + eps:
            key = (round(c, 12), round(r, 12))
            if key not in seen:
                seen.add(key)
                pts.append((c, r))

    if not pts:
        return 0.0

    # 按 cost 升序排列
    pts.sort(key=lambda x: (x[0], x[1]))

    hv = 0.0
    prev_r = rr
    for c, r in pts:
        width = max(0.0, rc - c)
        height = max(0.0, prev_r - r)
        hv += width * height
        prev_r = min(prev_r, r)

    return float(hv)
    

def run_once_spr_iter(
    instance_path: str,
    beta: float,
    run_id: int,
    seed: int,
    n_scenarios: int = 50,
    n_reduced_scenarios: int = 10,
    verbose: int = 1,
    log_every: int = 20,
    A_MAX: int = 60,
    max_iter: int = 200,
    patience: int = 50,
    max_eval: Optional[int] = None,
    eps=1e-9
):
    debug_log = []
    start_time = time.time()

    # =============================
    # 0. Problem
    # =============================
    problem = load_problem(instance_path, ev_params)
    instance_name = os.path.splitext(os.path.basename(instance_path))[0]

    # =============================
    # 1. Scenarios
    # =============================
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

    # =============================
    # 2. Evaluator / search objects
    # =============================
    rng = np.random.default_rng(seed)

    evaluator = Evaluator(problem, ra_safe=0.1, ra_risk=0.7)

    base_eval_fn = lambda routes: evaluator.evaluate_ffs(
        routes, reduced_scenarios, reduced_probs
    )
    cached_eval = EvalCache(base_eval_fn, max_size=30000)

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
        evaluator=cached_eval,
        eval_proxy=cached_eval,
        dominates_fn=dominates,
        rng=rng,
        eps=eps,
    )

    # =============================
    # 3. Clustering
    # =============================
    depot_coords = np.asarray(
        [(problem.nodes[i].x, problem.nodes[i].y) for i in problem.depots],
        dtype=float,
    )

    clusters = clustering(
        problem,
        method="kmeans",
        feature="location",
        random_state=seed,
    )

    # =============================
    # 4. GB + GA initialization
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
    # 5. Initial solution by FFS
    # =============================
    init_eval, routes_clusters = evaluator.evaluate_init_ffs(
        routes_clusters, reduced_scenarios, reduced_probs
    )
    init_sol = pack_solution(routes_clusters, init_eval)

    archive = []
    archive, _ = pareto_insert(archive, init_sol, eps=eps)
    if A_MAX is not None:
        archive = truncate_by_crowding(archive, max_size=A_MAX)

    cand_points = [(float(init_sol["cost"]), float(init_sol["ra"]))]

    print(init_sol["cost"], init_sol["ra"])

    # ===== 新增：参考点与收敛统计初始化 =====
    # 用初始解构造一个保守参考点；后面若遇到更差点再自动放大
    ref_cost = max(float(init_sol["cost"]) * 1.2, float(init_sol["cost"]) + 1.0)
    ref_ra = max(float(init_sol["ra"]) * 1.2, float(init_sol["ra"]) + 1e-6)

    archive_change_count = 0
    last_improve_iter = 0
    early_stop_triggered = False

    # =============================
    # 5. Multi-objective VNS + VND
    # =============================
    no_improve = 0
    k_max = len(NS)
    stop_by_eval = False

    for it in range(1, max_iter + 1):

        if max_eval is not None and cached_eval.miss >= max_eval:
            stop_by_eval = True
            if verbose >= 1:
                print(f"[Stop] reached max_eval={max_eval}.")
            break

        archive_changed = False
        iter_t0 = time.time()

        archive_size_before = len(archive)
        old_points_iter = {(float(s["cost"]), float(s["ra"])) for s in archive}

        k = 1
        while k <= k_max:

            if max_eval is not None and cached_eval.miss >= max_eval:
                stop_by_eval = True
                break

            old_points = {(float(s["cost"]), float(s["ra"])) for s in archive}

            # 1) MO-Shake
            archive_shake = mo_shake(
                archive=archive,
                shaker=shaker,
                evaluator=cached_eval,
                k=k,
                cid=None,
            )

            for sol in archive_shake:
                cand_points.append((float(sol["cost"]), float(sol["ra"])))

            # # 2) VND on cost
            # archive_vnd_cost = mo_vnd(
            #     shaken_set=archive_shake,
            #     vnd=vnd,
            #     obj="cost",
            #     NL=NL,
            #     cid=None,
            #     eps=eps,
            #     max_size=A_MAX,
            # )

            # # 3) VND on ra
            # archive_vnd_ra = mo_vnd(
            #     shaken_set=archive_vnd_cost,
            #     vnd=vnd,
            #     obj="ra",
            #     NL=NL,
            #     cid=None,
            #     eps=eps,
            #     max_size=A_MAX,
            # )

            # for sol in archive_vnd_ra:
            #     cand_points.append((float(sol["cost"]), float(sol["ra"])))

            #     # ===== 新增：动态更新 HV 参考点 =====
            #     ref_cost = max(ref_cost, float(sol["cost"]) * 1.05)
            #     ref_ra = max(ref_ra, float(sol["ra"]) * 1.05)

            # # 4) merge into archive
            # archive_new = list(archive)
            # for sol in archive_vnd_ra:
            #     archive_new, _ = pareto_insert(archive_new, sol, eps=eps)




            # 2) Alternate cost-first / ra-first by outer iteration
            if it % 2 == 1:
                first_obj, second_obj = "cost", "ra"
            else:
                first_obj, second_obj = "ra", "cost"

            archive_vnd_first = mo_vnd(
                shaken_set=archive_shake,
                vnd=vnd,
                obj=first_obj,
                NL=NL,
                cid=None,
                eps=eps,
                max_size=A_MAX,
            )

            archive_vnd_second = mo_vnd(
                shaken_set=archive_vnd_first,
                vnd=vnd,
                obj=second_obj,
                NL=NL,
                cid=None,
                eps=eps,
                max_size=A_MAX,
            )

            for sol in archive_vnd_second:
                cand_points.append((float(sol["cost"]), float(sol["ra"])))

                # 动态更新 HV 参考点
                ref_cost = max(ref_cost, float(sol["cost"]) * 1.05)
                ref_ra = max(ref_ra, float(sol["ra"]) * 1.05)

            # 4) merge into archive
            archive_new = list(archive)
            for sol in archive_vnd_second:
                archive_new, _ = pareto_insert(archive_new, sol, eps=eps)

            

            if A_MAX is not None:
                archive_new = truncate_by_crowding(archive_new, max_size=A_MAX)

            new_points = {(float(s["cost"]), float(s["ra"])) for s in archive_new}

            if new_points != old_points:
                archive = archive_new
                archive_changed = True
                k = 1
            else:
                k += 1
        
        if stop_by_eval:
            if verbose >= 1:
                print(f"[Stop] reached max_eval={max_eval}.")
            break

        # print(int(len(cand_points)))
        if archive_changed:
            no_improve = 0
            archive_change_count += 1
            last_improve_iter = it
        else:
            no_improve += 1
            if no_improve >= patience:
                early_stop_triggered = True
                if verbose >= 1:
                    print(f"[Stop] no improvement for {patience} outer iterations.")
                break

        iter_time = time.time() - iter_t0
        best_cost_sol = min(archive, key=lambda s: s["cost"])
        best_ra_sol = min(archive, key=lambda s: s["ra"])

        # ===== 新增：本代 archive 统计 =====
        arch_stat = archive_stats(archive)
        hv = hypervolume_2d_min(archive, ref_point=(ref_cost, ref_ra))

        new_points_iter = {(float(s["cost"]), float(s["ra"])) for s in archive}
        n_new_to_archive = len(new_points_iter - old_points_iter)
        n_removed_from_archive = len(old_points_iter - new_points_iter)

        if verbose >= 2 or (verbose >= 1 and (it % log_every) == 0):
            print(
                f"[Iter {it}] archive={len(archive)} "
                f"best_cost={best_cost_sol['cost']:.3f} "
                f"best_ra={best_ra_sol['ra']:.6f} "
                f"hv={hv:.6f} "
                f"changed={archive_changed}"
            )

        debug_log.append({
            "iter": int(it),

            "best_cost_in_archive": float(best_cost_sol["cost"]),
            "ra_of_best_cost": float(best_cost_sol["ra"]),
            "best_ra_in_archive": float(best_ra_sol["ra"]),
            "cost_of_best_ra": float(best_ra_sol["cost"]),

            "worst_cost_in_archive": float(arch_stat["worst_cost"]),
            "worst_ra_in_archive": float(arch_stat["worst_ra"]),
            "spread_cost": float(arch_stat["spread_cost"]),
            "spread_ra": float(arch_stat["spread_ra"]),

            "archive_size": int(len(archive)),
            "archive_changed": bool(archive_changed),
            "n_new_to_archive": int(n_new_to_archive),
            "n_removed_from_archive": int(n_removed_from_archive),

            "hv": float(hv),
            "hv_ref_cost": float(ref_cost),
            "hv_ref_ra": float(ref_ra),

            "no_improve": int(no_improve),
            "iter_time": float(iter_time),
            "n_candidates_so_far": int(len(cand_points)),
        })

    # =============================
    # 7. Final summary
    # =============================
    runtime = time.time() - start_time

    final_best = pick_work_grid(archive)
    if final_best is None:
        raise RuntimeError("Archive is empty at the end of search.")

    min_cost_sol = min(archive, key=lambda s: s["cost"])
    min_ra_sol = min(archive, key=lambda s: s["ra"])

    # ===== 新增：最终统计 =====
    final_arch_stat = archive_stats(archive)
    hv_final = hypervolume_2d_min(archive, ref_point=(ref_cost, ref_ra))
    cache_stats = cached_eval.stats()

    improve_cost_abs = float(init_sol["cost"] - min_cost_sol["cost"])
    improve_ra_abs = float(init_sol["ra"] - min_ra_sol["ra"])

    improve_cost_rel = (
        improve_cost_abs / float(init_sol["cost"])
        if abs(float(init_sol["cost"])) > 1e-12 else np.nan
    )
    improve_ra_rel = (
        improve_ra_abs / float(init_sol["ra"])
        if abs(float(init_sol["ra"])) > 1e-12 else np.nan
    )

    if verbose >= 1:
        print(
            f"[Final] final_best=(cost={final_best['cost']:.6f}, ra={final_best['ra']:.6f}) "
            f"best_cost={min_cost_sol['cost']:.6f} "
            f"best_ra={min_ra_sol['ra']:.6f} "
            f"archive={len(archive)} hv={hv_final:.6f} runtime={runtime:.2f}s"
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
        "n_iterations": int(len(debug_log)),

        # ===== 新增：收敛/质量摘要 =====
        "hv_final": float(hv_final),
        "hv_ref_cost": float(ref_cost),
        "hv_ref_ra": float(ref_ra),

        "spread_cost_final": float(final_arch_stat["spread_cost"]),
        "spread_ra_final": float(final_arch_stat["spread_ra"]),

        "archive_change_count": int(archive_change_count),
        "last_improve_iter": int(last_improve_iter),
        "stall_iters": int(no_improve),
        "early_stop_triggered": bool(early_stop_triggered),

        "improve_cost_abs": float(improve_cost_abs),
        "improve_cost_rel": float(improve_cost_rel),
        "improve_ra_abs": float(improve_ra_abs),
        "improve_ra_rel": float(improve_ra_rel),

        # ===== 新增：cache 统计 =====
        "eval_cache_hit": int(cache_stats["hit"]),
        "eval_cache_miss": int(cache_stats["miss"]),
        "eval_cache_hit_rate": float(cache_stats["hit_rate"]),
        "eval_cache_size": int(cache_stats["size"]),
        "eval_cache_max_size": int(cache_stats["max_size"]),

        "runtime_sec": float(runtime),
        "saa_S": int(scenarios.shape[0]),
        "n_reduced_scenarios": int(reduced_scenarios.shape[0]),

        "pareto_points": [(float(sol["cost"]), float(sol["ra"])) for sol in archive],
        "archive": archive,
        "final_best": final_best,
        "min_cost_sol": min_cost_sol,
        "min_ra_sol": min_ra_sol,
        "cand_points": cand_points,
        "clusters": clusters,
        "selected_idx": selected_idx,
        "debug_log": debug_log,

        "max_eval": None if max_eval is None else int(max_eval),
        "eval_count": int(cached_eval.miss),
        "stop_by_eval": bool(stop_by_eval),
    }

    return result



def run_once_spr(
    instance_path: str,
    beta: float,
    run_id: int,
    seed: int,
    n_scenarios: int = 50,
    n_reduced_scenarios: int = 10,
    verbose: int = 1,
    log_every: int = 20,
    A_MAX: int = 60,
    patience: Optional[int] = None,
    max_iter: Optional[int] = None,
    max_eval: Optional[int] = None,
    max_time_sec: Optional[float] = None,
    ra_safe_param: float = 0.1,
    ra_risk_param: float = 0.7,
    eps=1e-9
):
    # =============================
    # 0. Stop-condition validation
    # =============================
    if max_iter is not None and max_iter <= 0:
        raise ValueError("max_iter must be a positive integer or None.")
    if max_eval is not None and max_eval <= 0:
        raise ValueError("max_eval must be a positive integer or None.")
    if max_time_sec is not None and max_time_sec <= 0:
        raise ValueError("max_time_sec must be a positive number or None.")
    if patience is not None and patience <= 0:
        raise ValueError("patience must be a positive integer or None.")

    if all(x is None for x in [max_iter, max_eval, max_time_sec, patience]):
        raise ValueError("At least one stopping criterion must be set among max_iter, max_eval, max_time_sec, patience.")

    debug_log = []
    start_time = time.time()

    # =============================
    # 1. Problem
    # =============================
    problem = load_problem(instance_path, ev_params)
    instance_name = os.path.splitext(os.path.basename(instance_path))[0]

    # =============================
    # 2. Scenarios
    # =============================
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

    # =============================
    # 3. Evaluator / search objects
    # =============================
    rng = np.random.default_rng(seed)

    evaluator = Evaluator(problem, ra_safe=ra_safe_param, ra_risk=ra_risk_param)

    base_eval_fn = lambda routes: evaluator.evaluate_ffs(
        routes, reduced_scenarios, reduced_probs
    )
    cached_eval = EvalCache(base_eval_fn, max_size=30000)

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
        evaluator=cached_eval,
        eval_proxy=cached_eval,
        dominates_fn=dominates,
        rng=rng,
        eps=eps,
    )

    # =============================
    # 4. Clustering
    # =============================
    depot_coords = np.asarray(
        [(problem.nodes[i].x, problem.nodes[i].y) for i in problem.depots],
        dtype=float,
    )

    clusters = clustering(
        problem,
        method="kmeans",
        feature="location",
        random_state=seed,
    )

    # =============================
    # 5. GB + GA initialization
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
    # 6. Initial solution by FFS
    # =============================
    init_eval, routes_clusters = evaluator.evaluate_init_ffs(
        routes_clusters, reduced_scenarios, reduced_probs
    )
    init_sol = pack_solution(routes_clusters, init_eval)

    archive = []
    archive, _ = pareto_insert(archive, init_sol, eps=eps)
    if A_MAX is not None:
        archive = truncate_by_crowding(archive, max_size=A_MAX)

    cand_points = [(float(init_sol["cost"]), float(init_sol["ra"]))]

    if verbose >= 1:
        print(f"[Init] cost={init_sol['cost']:.6f}, ra={init_sol['ra']:.6f}")

    # 参考点
    ref_cost = max(float(init_sol["cost"]) * 1.2, float(init_sol["cost"]) + 1.0)
    ref_ra = max(float(init_sol["ra"]) * 1.2, float(init_sol["ra"]) + 1e-6)

    archive_change_count = 0
    last_improve_iter = 0
    early_stop_triggered = False

    # =============================
    # 7. Multi-objective VNS + VND
    # =============================
    no_improve = 0
    k_max = len(NS)

    stop_by_iter = False
    stop_by_eval = False
    stop_by_time = False
    stop_by_patience = False
    stop_reason = "not_stopped"

    it = 0
    while True:
        elapsed = time.time() - start_time

        if max_time_sec is not None and elapsed >= max_time_sec:
            stop_by_time = True
            stop_reason = "max_time"
            if verbose >= 1:
                print(f"[Stop] reached max_time_sec={max_time_sec}.")
            break

        if max_eval is not None and cached_eval.miss >= max_eval:
            stop_by_eval = True
            stop_reason = "max_eval"
            if verbose >= 1:
                print(f"[Stop] reached max_eval={max_eval}.")
            break

        if max_iter is not None and it >= max_iter:
            stop_by_iter = True
            stop_reason = "max_iter"
            if verbose >= 1:
                print(f"[Stop] reached max_iter={max_iter}.")
            break

        it += 1
        archive_changed = False
        iter_t0 = time.time()

        old_points_iter = {(float(s["cost"]), float(s["ra"])) for s in archive}

        k = 1
        while k <= k_max:
            elapsed = time.time() - start_time
            if max_time_sec is not None and elapsed >= max_time_sec:
                stop_by_time = True
                stop_reason = "max_time"
                break

            if max_eval is not None and cached_eval.miss >= max_eval:
                stop_by_eval = True
                stop_reason = "max_eval"
                break

            old_points = {(float(s["cost"]), float(s["ra"])) for s in archive}

            # 1) MO-Shake
            archive_shake = mo_shake(
                archive=archive,
                shaker=shaker,
                evaluator=cached_eval,
                k=k,
                cid=None,
            )

            for sol in archive_shake:
                cand_points.append((float(sol["cost"]), float(sol["ra"])))

            # 2) Alternate cost-first / ra-first by outer iteration
            if it % 2 == 1:
                first_obj, second_obj = "cost", "ra"
            else:
                first_obj, second_obj = "ra", "cost"

            archive_vnd_first = mo_vnd(
                shaken_set=archive_shake,
                vnd=vnd,
                obj=first_obj,
                NL=NL,
                cid=None,
                eps=eps,
                max_size=A_MAX,
            )

            elapsed = time.time() - start_time
            if max_time_sec is not None and elapsed >= max_time_sec:
                stop_by_time = True
                stop_reason = "max_time"
                break

            if max_eval is not None and cached_eval.miss >= max_eval:
                stop_by_eval = True
                stop_reason = "max_eval"
                break

            archive_vnd_second = mo_vnd(
                shaken_set=archive_vnd_first,
                vnd=vnd,
                obj=second_obj,
                NL=NL,
                cid=None,
                eps=eps,
                max_size=A_MAX,
            )

            for sol in archive_vnd_second:
                cand_points.append((float(sol["cost"]), float(sol["ra"])))
                ref_cost = max(ref_cost, float(sol["cost"]) * 1.05)
                ref_ra = max(ref_ra, float(sol["ra"]) * 1.05)

            archive_new = list(archive)
            for sol in archive_vnd_second:
                archive_new, _ = pareto_insert(archive_new, sol, eps=eps)

            if A_MAX is not None:
                archive_new = truncate_by_crowding(archive_new, max_size=A_MAX)

            new_points = {(float(s["cost"]), float(s["ra"])) for s in archive_new}

            if new_points != old_points:
                archive = archive_new
                archive_changed = True
                k = 1
            else:
                k += 1

        if stop_by_time or stop_by_eval or stop_by_iter:
            break

        if archive_changed:
            no_improve = 0
            archive_change_count += 1
            last_improve_iter = it
        else:
            no_improve += 1
            if patience is not None and no_improve >= patience:
                early_stop_triggered = True
                stop_by_patience = True
                stop_reason = "patience"
                if verbose >= 1:
                    print(f"[Stop] no improvement for {patience} outer iterations.")
                break

        iter_time = time.time() - iter_t0
        best_cost_sol = min(archive, key=lambda s: s["cost"])
        best_ra_sol = min(archive, key=lambda s: s["ra"])

        arch_stat = archive_stats(archive)
        hv = hypervolume_2d_min(archive, ref_point=(ref_cost, ref_ra))

        new_points_iter = {(float(s["cost"]), float(s["ra"])) for s in archive}
        n_new_to_archive = len(new_points_iter - old_points_iter)
        n_removed_from_archive = len(old_points_iter - new_points_iter)

        if verbose >= 2 or (verbose >= 1 and (it % log_every) == 0):
            print(
                f"[Iter {it}] archive={len(archive)} "
                f"best_cost={best_cost_sol['cost']:.3f} "
                f"best_ra={best_ra_sol['ra']:.6f} "
                f"hv={hv:.6f} "
                f"changed={archive_changed} "
                f"elapsed={time.time() - start_time:.2f}s"
            )

        debug_log.append({
            "iter": int(it),
            "elapsed_sec": float(time.time() - start_time),

            "best_cost_in_archive": float(best_cost_sol["cost"]),
            "ra_of_best_cost": float(best_cost_sol["ra"]),
            "best_ra_in_archive": float(best_ra_sol["ra"]),
            "cost_of_best_ra": float(best_ra_sol["cost"]),

            "worst_cost_in_archive": float(arch_stat["worst_cost"]),
            "worst_ra_in_archive": float(arch_stat["worst_ra"]),
            "spread_cost": float(arch_stat["spread_cost"]),
            "spread_ra": float(arch_stat["spread_ra"]),

            "archive_size": int(len(archive)),
            "archive_changed": bool(archive_changed),
            "n_new_to_archive": int(n_new_to_archive),
            "n_removed_from_archive": int(n_removed_from_archive),

            "hv": float(hv),
            "hv_ref_cost": float(ref_cost),
            "hv_ref_ra": float(ref_ra),

            "no_improve": int(no_improve),
            "iter_time": float(iter_time),
            "n_candidates_so_far": int(len(cand_points)),
        })

    # =============================
    # 8. Final summary
    # =============================
    runtime = time.time() - start_time

    min_cost_sol = min(archive, key=lambda s: s["cost"])
    min_ra_sol = min(archive, key=lambda s: s["ra"])

    init_stats = copy.deepcopy(init_sol.get("stats", {}))
    best_cost_stats = copy.deepcopy(
    min_cost_sol.get("stats", min_cost_sol.get("cend", {}).get("stats", {}))
    )
    best_ra_stats = copy.deepcopy(
        min_ra_sol.get("stats", min_ra_sol.get("cend", {}).get("stats", {}))
    )

    final_arch_stat = archive_stats(archive)
    hv_final = hypervolume_2d_min(archive, ref_point=(ref_cost, ref_ra))
    cache_stats = cached_eval.stats()

    improve_cost_abs = float(init_sol["cost"] - min_cost_sol["cost"])
    improve_ra_abs = float(init_sol["ra"] - min_ra_sol["ra"])

    improve_cost_rel = (
        improve_cost_abs / float(init_sol["cost"])
        if abs(float(init_sol["cost"])) > 1e-12 else np.nan
    )
    improve_ra_rel = (
        improve_ra_abs / float(init_sol["ra"])
        if abs(float(init_sol["ra"])) > 1e-12 else np.nan
    )

    if verbose >= 1:
        print(
            f"best_cost={min_cost_sol['cost']:.6f} "
            f"best_ra={min_ra_sol['ra']:.6f} "
            f"archive={len(archive)} hv={hv_final:.6f} runtime={runtime:.2f}s "
            f"stop_reason={stop_reason}"
        )
    print("init_sol keys:", init_sol.keys())
    print("min_cost_sol keys:", min_cost_sol.keys())
    print("min_ra_sol keys:", min_ra_sol.keys())
    print("archive[0] keys:", archive[0].keys())

        
    print("init_sol stats:", init_sol.get("stats"))
    print("min_cost_sol stats:", min_cost_sol.get("stats"))
    print("min_ra_sol stats:", min_ra_sol.get("stats"))

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "instance": instance_name,
        "beta": float(beta),
        "run_id": int(run_id),
        "seed": int(seed),
        "ra_safe_param": float(ra_safe_param),
        "ra_risk_param": float(ra_risk_param),

        "init_cost": float(init_sol["cost"]),
        "init_ra": float(init_sol["ra"]),

        "best_cost_in_archive": float(min_cost_sol["cost"]),
        "ra_of_best_cost": float(min_cost_sol["ra"]),

        "best_ra_in_archive": float(min_ra_sol["ra"]),
        "cost_of_best_ra": float(min_ra_sol["cost"]),

        "archive_size": int(len(archive)),
        "n_candidates": int(len(cand_points)),
        "n_iterations": int(len(debug_log)),

        "hv_final": float(hv_final),
        "hv_ref_cost": float(ref_cost),
        "hv_ref_ra": float(ref_ra),

        "spread_cost_final": float(final_arch_stat["spread_cost"]),
        "spread_ra_final": float(final_arch_stat["spread_ra"]),

        "archive_change_count": int(archive_change_count),
        "last_improve_iter": int(last_improve_iter),
        "stall_iters": int(no_improve),
        "early_stop_triggered": bool(early_stop_triggered),

        "improve_cost_abs": float(improve_cost_abs),
        "improve_cost_rel": float(improve_cost_rel),
        "improve_ra_abs": float(improve_ra_abs),
        "improve_ra_rel": float(improve_ra_rel),

        "eval_cache_hit": int(cache_stats["hit"]),
        "eval_cache_miss": int(cache_stats["miss"]),
        "eval_cache_hit_rate": float(cache_stats["hit_rate"]),
        "eval_cache_size": int(cache_stats["size"]),
        "eval_cache_max_size": int(cache_stats["max_size"]),

        "runtime_sec": float(runtime),
        "max_time_sec": None if max_time_sec is None else float(max_time_sec),
        "saa_S": int(scenarios.shape[0]),
        "n_reduced_scenarios": int(reduced_scenarios.shape[0]),

        "pareto_points": [(float(sol["cost"]), float(sol["ra"])) for sol in archive],
        "archive": archive,
        "min_cost_sol": min_cost_sol,
        "min_ra_sol": min_ra_sol,
        "cand_points": cand_points,
        "clusters": clusters,
        "selected_idx": selected_idx,
        "debug_log": debug_log,

        "max_iter_limit": None if max_iter is None else int(max_iter),
        "patience_limit": None if patience is None else int(patience),
        "max_eval": None if max_eval is None else int(max_eval),

        "eval_count": int(cached_eval.miss),
        "stop_by_iter": bool(stop_by_iter),
        "stop_by_eval": bool(stop_by_eval),
        "stop_by_time": bool(stop_by_time),
        "stop_by_patience": bool(stop_by_patience),
        "stop_reason": stop_reason,

        "init_stats": init_stats,
        "best_cost_stats": best_cost_stats,
        "best_ra_stats": best_ra_stats,
    }

    return result

# =========================================================
# 工具函数：把 numpy / pandas / 自定义对象尽量转成 json 可存格式
# =========================================================
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


# =========================================================
# 工具函数：从完整 result 提取适合写 progress CSV 的扁平记录
# =========================================================
def _flatten_stats_with_prefix(stats: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if stats is None:
        stats = {}

    keys = [
        "num_charge_total",
        "num_charge_opportunistic",
        "num_charge_risk",
        "num_charge_repair",
        "num_charge_natural",

        "charging_time_total",
        "charging_time_opportunistic",
        "charging_time_risk",
        "charging_time_repair",
        "charging_time_natural",

        "charged_energy_total",
        "charged_energy_opportunistic",
        "charged_energy_risk",
        "charged_energy_repair",
        "charged_energy_natural",
    ]

    out = {}
    for k in keys:
        out[f"{prefix}_{k}"] = stats.get(k, 0)
    return out

def _extract_progress_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    keep_cols = [
        "timestamp",
        "instance",
        "instance_path",
        "beta",
        "run_id",
        "seed",
        "ra_safe_param",
        "ra_risk_param",

        "init_cost",
        "init_ra",

        "best_cost_in_archive",
        "ra_of_best_cost",
        "best_ra_in_archive",
        "cost_of_best_ra",

        "archive_size",
        "n_candidates",
        "n_iterations",
        "hv_final",
        "hv_ref_cost",
        "hv_ref_ra",
        "spread_cost_final",
        "spread_ra_final",

        "archive_change_count",
        "last_improve_iter",
        "stall_iters",
        "early_stop_triggered",
        "stop_reason",
        "max_iter_limit",
        "patience_limit",

        "improve_cost_abs",
        "improve_cost_rel",
        "improve_ra_abs",
        "improve_ra_rel",

        "eval_cache_hit",
        "eval_cache_miss",
        "eval_cache_hit_rate",
        "eval_cache_size",
        "eval_cache_max_size",
        "eval_count",
        "max_eval",
        
        "stop_by_iter",
        "stop_by_eval",
        "stop_by_time",
        "stop_by_patience",

        "runtime_sec",
        "saa_S",
        "n_reduced_scenarios",
    ]

    out = {c: rec.get(c, None) for c in keep_cols}

    out.update(_flatten_stats_with_prefix(rec.get("best_cost_stats", {}), "best_cost"))
    out.update(_flatten_stats_with_prefix(rec.get("best_ra_stats", {}), "best_ra"))

    out["status"] = "completed"
    return out

# =========================================================
# 工具函数：列举 instances
# - 支持单文件
# - 支持文件夹
# - 支持递归扫描子文件夹
# =========================================================
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
        raise FileNotFoundError(f"No {suffix} instance files found under: {instance_source}")

    return files


# =========================================================
# 工具函数：任务唯一 key
# 统一用 instance 名称，而不是 path
# =========================================================
def _make_key(instance: str, beta: float, run_id: int):
    return (str(instance), round(float(beta), 6), int(run_id))

def _make_key_ra(instance: str, beta: float, ra_risk_param: float, run_id: int):
    return (
        str(instance),
        round(float(beta), 6),
        round(float(ra_risk_param), 6),
        int(run_id),
    )

# =========================================================
# 工具函数：安全追加一行 CSV
# =========================================================
def _append_row_csv(path: str, record: Dict[str, Any]):
    df_row = pd.DataFrame([record])
    write_header = not os.path.exists(path)
    df_row.to_csv(path, mode="a", header=write_header, index=False)


# =========================================================
# 工具函数：保存完整 json
# =========================================================
def _save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, ensure_ascii=False, indent=2)


# =========================================================
# 工具函数：从 progress 构建 summary
# =========================================================
def _build_summary(df: pd.DataFrame):
    required = {"instance", "beta"}
    if not required.issubset(df.columns):
        raise ValueError(f"_build_summary missing required columns: {required - set(df.columns)}")

    group_cols = ["instance", "beta"]
    if "ra_risk_param" in df.columns:
        group_cols.append("ra_risk_param")
    if "ra_safe_param" in df.columns:
        group_cols.append("ra_safe_param")

    agg_dict = {}

    def add_metric(col, ops):
        if col in df.columns:
            agg_dict[col] = ops

    add_metric("best_cost_in_archive", ["count", "mean", "std", "min", "median"])
    add_metric("best_ra_in_archive", ["mean", "std", "min", "median"])

    add_metric("runtime_sec", ["mean", "std", "min", "max", "median"])
    add_metric("hv_final", ["mean", "std", "max", "median"])
    add_metric("archive_size", ["mean", "std", "max", "median"])
    add_metric("n_iterations", ["mean", "std", "max", "median"])
    add_metric("n_candidates", ["mean", "std", "max", "median"])
    add_metric("eval_cache_hit_rate", ["mean", "std", "median"])
    add_metric("eval_count", ["mean", "std", "max", "median"])
    add_metric("improve_cost_rel", ["mean", "std", "max", "median"])
    add_metric("improve_ra_rel", ["mean", "std", "max", "median"])
    add_metric("t_stage1", ["mean", "std"])
    add_metric("t_stage2", ["mean", "std"])

    add_metric("best_cost_num_charge_total", ["mean"])
    add_metric("best_cost_num_charge_opportunistic", ["mean"])
    add_metric("best_cost_num_charge_risk", ["mean"])
    add_metric("best_cost_num_charge_repair", ["mean"])
    add_metric("best_cost_num_charge_natural", ["mean"])

    add_metric("best_cost_charging_time_total", ["mean"])
    add_metric("best_cost_charging_time_opportunistic", ["mean"])
    add_metric("best_cost_charging_time_risk", ["mean"])
    add_metric("best_cost_charging_time_repair", ["mean"])
    add_metric("best_cost_charging_time_natural", ["mean"])

    add_metric("best_cost_charged_energy_total", ["mean"])
    add_metric("best_cost_charged_energy_opportunistic", ["mean"])
    add_metric("best_cost_charged_energy_risk", ["mean"])
    add_metric("best_cost_charged_energy_repair", ["mean"])
    add_metric("best_cost_charged_energy_natural", ["mean"])

    add_metric("best_ra_num_charge_total", ["mean"])
    add_metric("best_ra_num_charge_opportunistic", ["mean"])
    add_metric("best_ra_num_charge_risk", ["mean"])
    add_metric("best_ra_num_charge_repair", ["mean"])
    add_metric("best_ra_num_charge_natural", ["mean"])

    add_metric("best_ra_charging_time_total", ["mean"])
    add_metric("best_ra_charging_time_opportunistic", ["mean"])
    add_metric("best_ra_charging_time_risk", ["mean"])
    add_metric("best_ra_charging_time_repair", ["mean"])
    add_metric("best_ra_charging_time_natural", ["mean"])

    add_metric("best_ra_charged_energy_total", ["mean"])
    add_metric("best_ra_charged_energy_opportunistic", ["mean"])
    add_metric("best_ra_charged_energy_risk", ["mean"])
    add_metric("best_ra_charged_energy_repair", ["mean"])
    add_metric("best_ra_charged_energy_natural", ["mean"])

    if len(agg_dict) == 0:
        return pd.DataFrame(columns=group_cols)

    summary = df.groupby(group_cols).agg(agg_dict)
    summary.columns = ["_".join(col).strip() for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()

    if "early_stop_triggered" in df.columns:
        tmp = df.groupby(group_cols)["early_stop_triggered"].mean().reset_index(name="early_stop_rate")
        summary = summary.merge(tmp, on=group_cols, how="left")

    if "stop_by_eval" in df.columns:
        tmp = df.groupby(group_cols)["stop_by_eval"].mean().reset_index(name="stop_by_eval_rate")
        summary = summary.merge(tmp, on=group_cols, how="left")

    if "t_stage2_mean" in summary.columns and "runtime_sec_mean" in summary.columns:
        summary["stage2_ratio_mean"] = summary["t_stage2_mean"] / summary["runtime_sec_mean"]

    return summary

def _worker_run_one(task):
    """
    子进程执行单个任务。
    返回统一结构，避免主进程不好处理。
    """
    (
        instance_path,
        instance_name,
        beta,
        run_id,
        seed,
        kwargs,
    ) = task

    try:
        # rec_full = run_once_spr(
        #     instance_path=instance_path,
        #     beta=beta,
        #     run_id=run_id,
        #     seed=seed,
        #     n_scenarios=kwargs["n_scenarios"],
        #     n_reduced_scenarios=kwargs["n_reduced_scenarios"],
        #     verbose=kwargs["verbose"],
        #     log_every=kwargs["log_every"],
        #     A_MAX=kwargs["A_MAX"],
        #     max_iter=kwargs["max_iter"],
        #     patience=kwargs["patience"],
        #     max_eval=kwargs.get("max_eval"),
        #     max_iter=kwargs.get("max_iter"),
        #     eps=kwargs["eps"],
        # )

        rec_full = run_once_spr(
            instance_path=instance_path,
            beta=beta,
            run_id=run_id,
            seed=seed,
            **kwargs,
        )

        rec_full["instance"] = rec_full.get("instance", instance_name)
        rec_full["instance_path"] = rec_full.get("instance_path", os.path.abspath(instance_path))
        rec_full["beta"] = float(rec_full.get("beta", beta))
        rec_full["run_id"] = int(rec_full.get("run_id", run_id))
        rec_full["seed"] = int(rec_full.get("seed", seed))

        return {
            "ok": True,
            "instance": instance_name,
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
            "beta": float(beta),
            "run_id": int(run_id),
            "seed": int(seed),
            "result": None,
            "error": repr(e),
        }
    


# =========================================================
# 主函数：可直接批量跑整个实例目录
# =========================================================
def run_experiments_resume_ultimate(
    instance_source: str,
    betas: List[float],
    n_runs: int,
    results_dir: Optional[str] = None,
    base_seed: int = 42,
    update_summary_each_run: bool = True,
    continue_on_error: bool = True,
    recursive: bool = True,
    instance_suffix: str = ".txt",

    # 透传给 run_once_spr 的参数
    n_scenarios: int = 100,
    n_reduced_scenarios: int = 10,
    verbose: int = 1,
    log_every: int = 20,
    A_MAX: int = 60,
    max_iter: int = 200,
    patience: int = 50,
    max_eval: Optional[int] = None,
    max_time_sec: float = 3600,
    eps: float = 1e-9,
):
    """
    适合论文实验的串行批量版：
    - 不依赖全局 BETAS / N_RUNS
    - 每个 run 自动保存 progress csv + iter csv + json
    - 支持断点续跑
    - 最终导出 runs / iter / summary / failed 到 Excel
    """
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    if results_dir is None:
        results_dir = os.path.join(os.getcwd(), "paper_experiment_results", run_tag)
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
        if {"instance", "beta", "run_id"}.issubset(df_done.columns):
            for _, row in df_done.iterrows():
                done.add(_make_key(row["instance"], row["beta"], row["run_id"]))
        print(f"[RESUME] Loaded progress: {progress_csv} | completed={len(done)}")
    else:
        print(f"[RESUME] No progress file. Will create: {progress_csv}")

    instance_list = _list_instances(instance_source=instance_source, recursive=recursive, suffix=instance_suffix)

    tasks = []
    total_tasks = 0
    skipped = 0
    for instance_path in instance_list:
        instance_name = os.path.splitext(os.path.basename(instance_path))[0]
        for beta in betas:
            for run_id in range(1, n_runs + 1):
                total_tasks += 1
                key = _make_key(instance_name, beta, run_id)
                if key in done:
                    skipped += 1
                    continue
                seed = int(base_seed + 1000 * round(float(beta) * 10) + (run_id - 1))
                tasks.append((instance_path, instance_name, float(beta), int(run_id), seed))

    print(f"[INSTANCES] found={len(instance_list)}")
    print(f"[TASKS] total={total_tasks} | skipped(done)={skipped} | remaining={len(tasks)}")

    if len(tasks) == 0:
        if os.path.exists(progress_csv):
            df_all = pd.read_csv(progress_csv)
            df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
            summary = _build_summary(df_ok)
            with pd.ExcelWriter(final_xlsx, engine="openpyxl") as writer:
                df_all.to_excel(writer, sheet_name="runs", index=False)
                if os.path.exists(iter_csv):
                    pd.read_csv(iter_csv).to_excel(writer, sheet_name="iter_log", index=False)
                summary.to_excel(writer, sheet_name="summary", index=False)
                if os.path.exists(failed_csv):
                    pd.read_csv(failed_csv).to_excel(writer, sheet_name="failed", index=False)
            print(f"[DONE] Nothing to run. Final Excel refreshed: {final_xlsx}")
        else:
            print("[DONE] No tasks and no progress file.")
        return

    for idx, (instance_path, instance_name, beta, run_id, seed) in enumerate(tasks, 1):
        print(f"\n=== [{idx}/{len(tasks)}] instance={instance_name} beta={beta:.3f} run={run_id}/{n_runs} seed={seed} ===")
        try:
            rec_full = run_once_spr(
                instance_path=instance_path,
                beta=beta,
                run_id=run_id,
                seed=seed,
                n_scenarios=n_scenarios,
                n_reduced_scenarios=n_reduced_scenarios,
                verbose=verbose,
                log_every=log_every,
                A_MAX=A_MAX,
                max_iter=max_iter,
                max_time_sec=max_time_sec,
                patience=patience,
                max_eval=max_eval,
                eps=eps,
            )

            rec_full["instance"] = rec_full.get("instance", instance_name)
            rec_full["instance_path"] = rec_full.get("instance_path", os.path.abspath(instance_path))
            rec_full["beta"] = float(rec_full.get("beta", beta))
            rec_full["run_id"] = int(rec_full.get("run_id", run_id))
            rec_full["seed"] = int(rec_full.get("seed", seed))

            json_name = f"{instance_name}_beta{beta:.3f}_run{run_id}.json".replace("/", "_")
            json_path = os.path.join(detail_dir, json_name)
            _save_json(json_path, rec_full)

            rec_flat = _extract_progress_record(rec_full)
            _append_row_csv(progress_csv, rec_flat)
            append_debug_log_to_iter_csv(rec_full, iter_csv)
            done.add(_make_key(rec_flat["instance"], rec_flat["beta"], rec_flat["run_id"]))

            if update_summary_each_run:
                df_all = pd.read_csv(progress_csv)
                df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
                _build_summary(df_ok).to_csv(summary_live_csv, index=False)

        except Exception as e:
            err = repr(e)
            print(f"[ERROR] instance={instance_name} beta={beta} run={run_id} failed: {err}")
            fail_rec = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "instance": instance_name,
                "instance_path": os.path.abspath(instance_path),
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

    if os.path.exists(progress_csv):
        df_all = pd.read_csv(progress_csv)
        df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
        summary = _build_summary(df_ok)
        with pd.ExcelWriter(final_xlsx, engine="openpyxl") as writer:
            df_all.to_excel(writer, sheet_name="runs", index=False)
            if os.path.exists(iter_csv):
                pd.read_csv(iter_csv).to_excel(writer, sheet_name="iter_log", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
            if os.path.exists(failed_csv):
                pd.read_csv(failed_csv).to_excel(writer, sheet_name="failed", index=False)

        print(f"\n[DONE] Progress CSV : {progress_csv}")
        if update_summary_each_run:
            print(f"[DONE] Live Summary : {summary_live_csv}")
        if os.path.exists(failed_csv):
            print(f"[DONE] Failed Runs  : {failed_csv}")
        print(f"[DONE] Iter Log CSV : {iter_csv}")
        print(f"[DONE] Detail JSONs  : {detail_dir}")
        print(f"[DONE] Final Excel   : {final_xlsx}")
    else:
        print("[DONE] No progress file found; nothing to export.")


def run_experiments_resume_parallel(
    instance_source: str,
    betas: List[float],
    n_runs: int,
    results_dir: Optional[str] = None,
    base_seed: int = 42,
    max_workers: int = 10,
    update_summary_each_run: bool = True,
    continue_on_error: bool = True,
    recursive: bool = True,
    instance_suffix: str = ".txt",

    # 透传给 run_once_spr 的参数
    n_scenarios: int = 50,
    n_reduced_scenarios: int = 10,
    verbose: int = 1,
    log_every: int = 20,
    A_MAX: int = 60,
    max_iter: int = 200,
    patience: int = 50,
    max_eval: Optional[int] = None,
    max_time_sec: float = 3600,
    eps: float = 1e-9,

    ra_safe_param: float = 0.1,
    ra_risk_param: float = 0.7,
):
    """
    适合论文实验的并行批量版：
    - 不依赖全局 BETAS / N_RUNS
    - 主进程统一写 progress csv / iter csv / json / excel
    - 支持断点续跑
    """
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    if results_dir is None:
        results_dir = os.path.join(os.getcwd(), "paper_experiment_results_parallel", run_tag)
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
        if {"instance", "beta", "run_id"}.issubset(df_done.columns):
            for _, row in df_done.iterrows():
                done.add(_make_key(row["instance"], row["beta"], row["run_id"]))
        print(f"[RESUME] Loaded progress: {progress_csv} | completed={len(done)}")
    else:
        print(f"[RESUME] No progress file. Will create: {progress_csv}")

    instance_list = _list_instances(instance_source=instance_source, recursive=recursive, suffix=instance_suffix)
    common_kwargs = {
        "n_scenarios": n_scenarios,
        "n_reduced_scenarios": n_reduced_scenarios,
        "verbose": verbose,
        "log_every": log_every,
        "A_MAX": A_MAX,
        "max_iter": max_iter,
        "patience": patience,
        "max_eval": max_eval,
        "max_time_sec": max_time_sec,
        "eps": eps,
        "ra_safe_param": ra_safe_param,
        "ra_risk_param": ra_risk_param,
    }

    tasks = []
    total_tasks = 0
    skipped = 0
    for instance_path in instance_list:
        instance_name = os.path.splitext(os.path.basename(instance_path))[0]
        for beta in betas:
            for run_id in range(1, n_runs + 1):
                total_tasks += 1
                key = _make_key(instance_name, beta, run_id)
                if key in done:
                    skipped += 1
                    continue
                seed = int(base_seed + 1000 * round(float(beta) * 10) + (run_id - 1))
                tasks.append((instance_path, instance_name, float(beta), int(run_id), seed, common_kwargs))

    print(f"[INSTANCES] found={len(instance_list)}")
    print(f"[TASKS] total={total_tasks} | skipped(done)={skipped} | remaining={len(tasks)}")
    print(f"[PARALLEL] max_workers={max_workers}")

    if len(tasks) == 0:
        if os.path.exists(progress_csv):
            df_all = pd.read_csv(progress_csv)
            df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
            summary = _build_summary(df_ok)
            with pd.ExcelWriter(final_xlsx, engine="openpyxl") as writer:
                df_all.to_excel(writer, sheet_name="runs", index=False)
                if os.path.exists(iter_csv):
                    pd.read_csv(iter_csv).to_excel(writer, sheet_name="iter_log", index=False)
                summary.to_excel(writer, sheet_name="summary", index=False)
                if os.path.exists(failed_csv):
                    pd.read_csv(failed_csv).to_excel(writer, sheet_name="failed", index=False)
            print(f"[DONE] Nothing to run. Final Excel refreshed: {final_xlsx}")
        else:
            print("[DONE] No tasks and no progress file.")
        return

    n_ok = 0
    n_fail = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(_worker_run_one, task): task for task in tasks}
        for i, future in enumerate(as_completed(future_to_task), 1):
            res = future.result()
            if res["ok"]:
                rec_full = res["result"]
                json_name = f"{rec_full['instance']}_beta{rec_full['beta']:.3f}_run{rec_full['run_id']}.json".replace("/", "_")
                json_path = os.path.join(detail_dir, json_name)
                _save_json(json_path, rec_full)

                rec_flat = _extract_progress_record(rec_full)
                _append_row_csv(progress_csv, rec_flat)
                append_debug_log_to_iter_csv(rec_full, iter_csv)
                done.add(_make_key(rec_flat["instance"], rec_flat["beta"], rec_flat["run_id"]))
                n_ok += 1

                print(
                    f"[OK {i}/{len(tasks)}] instance={rec_flat['instance']} beta={rec_flat['beta']:.3f} "
                    f"run={rec_flat['run_id']} best_cost={rec_flat['best_cost_in_archive']:.6f} "
                    f"best_ra={rec_flat['best_ra_in_archive']:.6f}"
                )

                if update_summary_each_run:
                    df_all = pd.read_csv(progress_csv)
                    df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
                    _build_summary(df_ok).to_csv(summary_live_csv, index=False)
            else:
                n_fail += 1
                fail_rec = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "instance": res["instance"],
                    "instance_path": res["instance_path"],
                    "beta": float(res["beta"]),
                    "run_id": int(res["run_id"]),
                    "seed": int(res["seed"]),
                    "status": "failed",
                    "error": res["error"],
                }
                _append_row_csv(failed_csv, fail_rec)
                print(f"[FAIL {i}/{len(tasks)}] instance={res['instance']} beta={res['beta']:.3f} run={res['run_id']} error={res['error']}")
                if not continue_on_error:
                    raise RuntimeError(res["error"])

    if os.path.exists(progress_csv):
        df_all = pd.read_csv(progress_csv)
        df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
        summary = _build_summary(df_ok)
        with pd.ExcelWriter(final_xlsx, engine="openpyxl") as writer:
            df_all.to_excel(writer, sheet_name="runs", index=False)
            if os.path.exists(iter_csv):
                pd.read_csv(iter_csv).to_excel(writer, sheet_name="iter_log", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
            if os.path.exists(failed_csv):
                pd.read_csv(failed_csv).to_excel(writer, sheet_name="failed", index=False)

        print(f"\n[DONE] success={n_ok}, failed={n_fail}")
        print(f"[DONE] Progress CSV : {progress_csv}")
        if update_summary_each_run:
            print(f"[DONE] Live Summary : {summary_live_csv}")
        if os.path.exists(failed_csv):
            print(f"[DONE] Failed Runs  : {failed_csv}")
        print(f"[DONE] Iter Log CSV : {iter_csv}")
        print(f"[DONE] Detail JSONs  : {detail_dir}")
        print(f"[DONE] Final Excel   : {final_xlsx}")
    else:
        print("[DONE] No progress file found; nothing to export.")


# =========================================================
# RA 敏感度分享
# =========================================================
def _make_key_ra2(instance: str, beta: float, ra_safe_param: float, ra_risk_param: float, run_id: int):
    return (
        str(instance),
        round(float(beta), 6),
        round(float(ra_safe_param), 6),
        round(float(ra_risk_param), 6),
        int(run_id),
    )

def _worker_run_one_ra2(task):
    (
        instance_path,
        instance_name,
        beta,
        ra_safe_param,
        ra_risk_param,
        run_id,
        seed,
        kwargs,
    ) = task

    try:
        rec_full = run_once_spr(
            instance_path=instance_path,
            beta=beta,
            run_id=run_id,
            seed=seed,
            n_scenarios=kwargs["n_scenarios"],
            n_reduced_scenarios=kwargs["n_reduced_scenarios"],
            verbose=kwargs["verbose"],
            log_every=kwargs["log_every"],
            A_MAX=kwargs["A_MAX"],
            patience=kwargs["patience"],
            max_eval=kwargs.get("max_eval"),
            max_iter=kwargs.get("max_iter"),
            max_time_sec=kwargs["max_time_sec"],
            ra_safe_param=ra_safe_param,
            ra_risk_param=ra_risk_param,
            eps=kwargs["eps"],
        )

        rec_full["instance"] = rec_full.get("instance", instance_name)
        rec_full["instance_path"] = rec_full.get("instance_path", os.path.abspath(instance_path))
        rec_full["beta"] = float(rec_full.get("beta", beta))
        rec_full["run_id"] = int(rec_full.get("run_id", run_id))
        rec_full["seed"] = int(rec_full.get("seed", seed))
        rec_full["ra_safe_param"] = float(rec_full.get("ra_safe_param", ra_safe_param))
        rec_full["ra_risk_param"] = float(rec_full.get("ra_risk_param", ra_risk_param))

        return {
            "ok": True,
            "instance": instance_name,
            "beta": float(beta),
            "ra_safe_param": float(ra_safe_param),
            "ra_risk_param": float(ra_risk_param),
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
            "beta": float(beta),
            "ra_safe_param": float(ra_safe_param),
            "ra_risk_param": float(ra_risk_param),
            "run_id": int(run_id),
            "seed": int(seed),
            "result": None,
            "error": repr(e),
        }

def run_ra_grid_experiments_resume_parallel(
    instance_source: str,
    betas: List[float],
    ra_safe_list: List[float],
    ra_risk_list: List[float],
    n_runs: int,
    results_dir: Optional[str] = None,
    base_seed: int = 42,
    max_workers: int = 10,
    update_summary_each_run: bool = True,
    continue_on_error: bool = True,
    recursive: bool = True,
    instance_suffix: str = ".txt",

    n_scenarios: int = 50,
    n_reduced_scenarios: int = 10,
    verbose: int = 1,
    log_every: int = 20,
    A_MAX: int = 60,
    max_iter: int = 200,
    patience: int = 50,
    max_eval: Optional[int] = None,
    max_time_sec: float = 3600,
    eps: float = 1e-9,

    append_run_tag: bool = False,
):
    if results_dir is None:
        results_dir = os.path.join(os.getcwd(), "paper_ra_grid_results")
    if append_run_tag:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        if {"instance", "beta", "ra_safe_param", "ra_risk_param", "run_id"}.issubset(df_done.columns):
            for _, row in df_done.iterrows():
                done.add(
                    _make_key_ra2(
                        row["instance"],
                        row["beta"],
                        row["ra_safe_param"],
                        row["ra_risk_param"],
                        row["run_id"],
                    )
                )
        print(f"[RESUME] Loaded progress: {progress_csv} | completed={len(done)}")
    else:
        print(f"[RESUME] No progress file. Will create: {progress_csv}")

    instance_list = _list_instances(instance_source=instance_source, recursive=recursive, suffix=instance_suffix)

    common_kwargs = {
        "n_scenarios": n_scenarios,
        "n_reduced_scenarios": n_reduced_scenarios,
        "verbose": verbose,
        "log_every": log_every,
        "A_MAX": A_MAX,
        "max_iter": max_iter,
        "patience": patience,
        "max_eval": max_eval,
        "max_time_sec": max_time_sec,
        "eps": eps,
    }

    tasks = []
    total_tasks = 0
    skipped = 0

    for instance_path in instance_list:
        instance_name = os.path.splitext(os.path.basename(instance_path))[0]
        for beta in betas:
            for ra_safe_param in ra_safe_list:
                for ra_risk_param in ra_risk_list:
                    if float(ra_risk_param) <= float(ra_safe_param):
                        continue

                    for run_id in range(1, n_runs + 1):
                        total_tasks += 1
                        key = _make_key_ra2(instance_name, beta, ra_safe_param, ra_risk_param, run_id)
                        if key in done:
                            skipped += 1
                            continue

                        seed = int(base_seed + 1000 * round(float(beta) * 10) + (run_id - 1))

                        tasks.append((
                            instance_path,
                            instance_name,
                            float(beta),
                            float(ra_safe_param),
                            float(ra_risk_param),
                            int(run_id),
                            seed,
                            common_kwargs,
                        ))

    print(f"[INSTANCES] found={len(instance_list)}")
    print(f"[TASKS] total={total_tasks} | skipped(done)={skipped} | remaining={len(tasks)}")
    print(f"[PARALLEL] max_workers={max_workers}")

    if len(tasks) == 0:
        if os.path.exists(progress_csv):
            df_all = pd.read_csv(progress_csv)
            df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
            summary = _build_summary(df_ok)
            with pd.ExcelWriter(final_xlsx, engine="openpyxl") as writer:
                df_all.to_excel(writer, sheet_name="runs", index=False)
                if os.path.exists(iter_csv):
                    pd.read_csv(iter_csv).to_excel(writer, sheet_name="iter_log", index=False)
                summary.to_excel(writer, sheet_name="summary", index=False)
                if os.path.exists(failed_csv):
                    pd.read_csv(failed_csv).to_excel(writer, sheet_name="failed", index=False)
            print(f"[DONE] Nothing to run. Final Excel refreshed: {final_xlsx}")
        else:
            print("[DONE] No tasks and no progress file.")
        return

    n_ok = 0
    n_fail = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(_worker_run_one_ra2, task): task for task in tasks}

        for i, future in enumerate(as_completed(future_to_task), 1):
            res = future.result()

            if res["ok"]:
                rec_full = res["result"]

                json_name = (
                    f"{rec_full['instance']}"
                    f"_beta{rec_full['beta']:.3f}"
                    f"_safe{rec_full['ra_safe_param']:.2f}"
                    f"_risk{rec_full['ra_risk_param']:.2f}"
                    f"_run{rec_full['run_id']}.json"
                ).replace("/", "_")
                json_path = os.path.join(detail_dir, json_name)
                _save_json(json_path, rec_full)

                rec_flat = _extract_progress_record(rec_full)
                _append_row_csv(progress_csv, rec_flat)
                append_debug_log_to_iter_csv(rec_full, iter_csv)

                done.add(_make_key_ra2(
                    rec_flat["instance"],
                    rec_flat["beta"],
                    rec_flat["ra_safe_param"],
                    rec_flat["ra_risk_param"],
                    rec_flat["run_id"],
                ))
                n_ok += 1

                print(
                    f"[OK {i}/{len(tasks)}] "
                    f"instance={rec_flat['instance']} "
                    f"beta={rec_flat['beta']:.3f} "
                    f"safe={rec_flat['ra_safe_param']:.2f} "
                    f"risk={rec_flat['ra_risk_param']:.2f} "
                    f"run={rec_flat['run_id']} "
                    f"best_cost={rec_flat['best_cost_in_archive']:.6f} "
                    f"best_ra={rec_flat['best_ra_in_archive']:.6f}"
                )

                if update_summary_each_run:
                    df_all = pd.read_csv(progress_csv)
                    df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
                    _build_summary(df_ok).to_csv(summary_live_csv, index=False)

            else:
                n_fail += 1
                fail_rec = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "instance": res["instance"],
                    "instance_path": res["instance_path"],
                    "beta": float(res["beta"]),
                    "ra_safe_param": float(res["ra_safe_param"]),
                    "ra_risk_param": float(res["ra_risk_param"]),
                    "run_id": int(res["run_id"]),
                    "seed": int(res["seed"]),
                    "status": "failed",
                    "error": res["error"],
                }
                _append_row_csv(failed_csv, fail_rec)

                print(
                    f"[FAIL {i}/{len(tasks)}] "
                    f"instance={res['instance']} "
                    f"beta={res['beta']:.3f} "
                    f"safe={res['ra_safe_param']:.2f} "
                    f"risk={res['ra_risk_param']:.2f} "
                    f"run={res['run_id']} "
                    f"error={res['error']}"
                )

                if not continue_on_error:
                    raise RuntimeError(res["error"])

    if os.path.exists(progress_csv):
        df_all = pd.read_csv(progress_csv)
        df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
        summary = _build_summary(df_ok)

        with pd.ExcelWriter(final_xlsx, engine="openpyxl") as writer:
            df_all.to_excel(writer, sheet_name="runs", index=False)
            if os.path.exists(iter_csv):
                pd.read_csv(iter_csv).to_excel(writer, sheet_name="iter_log", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
            if os.path.exists(failed_csv):
                pd.read_csv(failed_csv).to_excel(writer, sheet_name="failed", index=False)

        print(f"\n[DONE] success={n_ok}, failed={n_fail}")
        print(f"[DONE] Progress CSV : {progress_csv}")
        if update_summary_each_run:
            print(f"[DONE] Live Summary : {summary_live_csv}")
        if os.path.exists(failed_csv):
            print(f"[DONE] Failed Runs  : {failed_csv}")
        print(f"[DONE] Iter Log CSV : {iter_csv}")
        print(f"[DONE] Detail JSONs  : {detail_dir}")
        print(f"[DONE] Final Excel   : {final_xlsx}")
    else:
        print("[DONE] No progress file found; nothing to export.")
        


        
if __name__ == "__main__":

    ## ========= 单个文件运行，调试
    instance_name = 'c101_21'
    base_path = r"D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)"
    file_path = os.path.join(base_path, f"{instance_name}.txt")

    res = run_once_spr(
        instance_path=file_path,
        beta=0.2,
        run_id=1,
        seed=42,
        n_scenarios=100,
        n_reduced_scenarios=10,
        verbose=1,
        log_every=1,
        A_MAX= 60,
        patience= None,
        max_iter= None,
        max_eval= 100000,
        max_time_sec=200000,
        ra_safe_param=0.1,
        ra_risk_param= 0.7,
        eps=1e-9
    )

    archive = res["archive"]
    cand_points = res["cand_points"]

    print(res["runtime_sec"])
    print(res["archive_size"])

    plot_pareto_front(
        archive=archive,
        title=f"Pareto Front - {instance_name}"
    )

    plot_search_evolution(
        cand_points=cand_points,
        archive=archive,
        title=f"Search Evolution - {instance_name}"
    )


    log_df = pd.DataFrame(res["debug_log"])

    log_df.plot(x="iter", y="hv", title="HV vs Iteration")
    log_df.plot(x="iter", y="archive_size", title="Archive Size vs Iteration")
    # log_df.plot(x="iter", y=["best_cost_in_archive", "best_ra_in_archive"], title="Best Objectives vs Iteration")
    log_df.plot(x="iter", y="best_cost_in_archive", title="Best Objectives vs Iteration")
    log_df.plot(x="iter", y="best_ra_in_archive", title="Best Objectives vs Iteration")

    plt.show()




    # # # =========================
    # # # 你自己的全局参数
    # # # =========================
    # BETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # # betas = [0.2]
    # n_runs = 10

    # ## ======== 整个文件夹运行
    # path = r"D:\02_Research\DataSet\Paper_MO_SPR\Sensitivity"
    # out_path = r"D:\02_Research\6_experimentResults\Sensitivity\Beta"

    # run_experiments_resume_parallel(
    #     instance_source=path,
    #     betas=BETAS,
    #     n_runs=n_runs,
    #     results_dir=out_path,
    #     base_seed=42,
    #     max_workers=10,
    #     recursive=True,
    #     n_scenarios=100,
    #     n_reduced_scenarios=10,
    #     max_time_sec=1200,   # 1小时
    #     # patience=None,
    #     max_eval=None,
    #     verbose=1,
    #     ra_safe_param=0.1,
    #     ra_risk_param= 0.7,
    # )



    # step = 0.05
    # ra_safe_list = [round(i * step, 2) for i in range(21)]   # 0.00, 0.05, ..., 1.00
    # ra_risk_list = [round(i * step, 2) for i in range(21)]   # 0.00, 0.05, ..., 1.00

    # path = r"D:\02_Research\DataSet\Paper_MO_SPR\Sensitivity"
    # out_path = r"D:\02_Research\6_experimentResults\Sensitivity"

    # run_ra_grid_experiments_resume_parallel(
    #     instance_source=path,
    #     betas=betas,
    #     ra_safe_list=ra_safe_list,
    #     ra_risk_list=ra_risk_list,
    #     n_runs=n_runs,
    #     results_dir=out_path,
    #     base_seed=42,
    #     max_workers=10,
    #     recursive=True,
    #     n_scenarios=100,
    #     n_reduced_scenarios=10,
    #     max_time_sec=1200,
    #     max_iter = 100,
    #     patience = 20,
    #     max_eval=None,
    #     verbose=1,
    #     append_run_tag=False,
    # )

