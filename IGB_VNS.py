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
from plot import plot_clusters_gbs, plot_clusters_gbs_with_gb_routes, plot_routes
from Modules.VNS.neighborhood import Neighborhood
from Modules.VNS.shaking import Shaker
from Modules.VNS.vnd import VNDRefiner



def pack_solution(routes_by_clusters, cend):
    return {
        "routes_by_clusters": copy.deepcopy(routes_by_clusters),
        "cost": float(cend["total_cost"]),
        "ra": float(cend.get("total_ra", 0.0)),
        "cend": cend,
    }

def dominates_cost(a, b, eps=1e-9):
    return a["cost"] < b["cost"] - eps

def flatten_routes(routes_clusters):
    return {cid: [n for r in routes for n in r] for cid, routes in routes_clusters.items()}

def assert_routes_ok(routes_clusters, expected_set, stage=""):
    flat = flatten_routes(routes_clusters)
    for cid, seq in flat.items():
        exp = expected_set[cid]
        outsiders = [n for n in seq if n not in exp]
        missing = sorted(exp - set(seq))
        cnt = Counter(seq)
        dup = sorted([n for n, c in cnt.items() if c > 1])

        if outsiders or missing or dup:
            print("\n[ROUTE CHECK FAILED]", stage, "cid=", cid)
            print("len(seq)=", len(seq), "unique=", len(set(seq)), "expected=", len(exp))
            if outsiders:
                print("outsiders:", outsiders[:30])
            if missing:
                print("missing:", missing[:30])
            if dup:
                print("duplicates:", dup[:30])
                print("dup counts:", [(n, cnt[n]) for n in dup[:10]])
            raise AssertionError("routes integrity failed")



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


def main():
    # -----------------------------
    # 1. Load instance
    # -----------------------------
    instance_name = 'rc103_21'
    base_path = r"D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)"
    file_path = os.path.join(base_path, f"{instance_name}.txt")

    ev_params = {
        "soc_max": 40.0,
        "vehicle_capacity": 650.0,
    }

    problem = load_problem(file_path, ev_params)
    evaluator = Evaluator(problem, ra_safe=0.1, ra_risk=0.7)

    # print("Loaded problem:")
    # print("customers =", len(problem.customers))
    # print("stations  =", len(problem.css))
    # print("depots    =", len(problem.depots))

    # mu: 每个客户的期望需求（通常就用实例中的确定性 demand 当作 mu）
    mu = np.asarray(problem.demand, dtype=float)
    beta = 0.1
    scenarios = generate_normal_scenarios(
        mu=mu,
        beta=beta,
        n_scenarios=100,
        random_state=100
    )


    # # -----------------------------
    # # 2. Run clustering (EVRP API)
    # # -----------------------------
    clusters = clustering(
        problem,
        method="kmeans",          # kmeans / kmedoids
        feature="location",         # location / time_window
        random_state=1,
    )

    # clusters ={
    # 0: {22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 96, 106, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121}, 
    # 1: {41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 97}, 
    # 2: {74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,  98, 99, 100, 101, 102, 103, 104, 105, 107, 108}
    # }
    # # {
    # # 0: {22, 24, 30, 31, 32, 40, 41, 48, 51, 52, 53, 54, 55, 56, 70, 71, 72, 83, 84, 85, 86, 87, 90, 91, 92, 97, 98, 99, 100, 102, 109, 111}, 
    # # 1: {26, 27, 28, 29, 34, 35, 37, 38, 39, 57, 58, 59, 63, 64, 65, 66, 67, 68, 69, 73, 80, 81, 82, 103, 104, 105, 106, 107, 108, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121}, 
    # # 2: {23, 25, 33, 36, 42, 43, 44, 45, 46, 47, 49, 50, 60, 61, 62, 74, 75, 76, 77, 78, 79, 88, 89, 93, 94, 95, 96, 101}
    # # }

    # # clusters: dict[int, set[int]]
    # print("\nClustering result:")
    # print("n_clusters =", len(clusters))

    # for cid, members in clusters.items():
    #     print(f"Cluster {cid}: size={len(members)}")
    # print(clusters)

    # -----------------------------
    # 3. Plot clustering result
    # -----------------------------

    # plot_clusters(problem, clusters, feature="time_window")
    # plot_clusters(problem, clusters, feature="location")
    # plot_clusters(problem, clusters, feature="time_window_mid_width")

    start_time = time.time()


    depot_coords = np.asarray(
        [(problem.nodes[i].x, problem.nodes[i].y) for i in problem.depots], dtype=float
    )
    
    # ========== 1) 初始化：GB-ACO ==========
    gbs_by_cluster = {}        # cid -> gbs_customer_id (按 gb_id)
    gb_order_by_cluster = {}   # cid -> gb_order (pos->gb_id)

    routes_clusters = {}    # cid -> routes (按 pos)
    for cid, cluster in clusters.items():
        customer_ids = list(cluster)
        points = np.asarray([(problem.nodes[i].x, problem.nodes[i].y) for i in customer_ids], dtype=float)

        # ===== 生成 gbs =====
        gbs_info = gb_clustering(
            points,
            method="gb_kmeans",
            split_k=0.6,
            merge=True,
            random_state=None,
        )

        gbs_customer_id = [[customer_ids[j] for j in cl] for cl in gbs_info.gbs]
        gbs_by_cluster[cid] = gbs_customer_id

        # # ===== 规划 gbs 之间的路径 =====
        # params_GB = ACOParams(num_ants=10, num_iter=60)
        # gb_order_result = plan_gb_order(gbs_info.centers, depot_coords, params_GB)

        # gb_order_by_cluster[cid] = gb_order_result.order

        # # ===== 规划 gb 内客户的路径  =====
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
        gb_order_by_cluster[cid] = gb_order_result.order

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


    # #  === plot gbs 以及 gbs 之间的路径
    # plot_clusters_gbs(problem, routes_clusters)
    # cluster_keys = sorted(gbs_by_cluster.keys())
    # gb_routes_list = [gb_order_by_cluster[cid] for cid in cluster_keys]
    # plot_clusters_gbs_with_gb_routes(problem, gbs_by_cluster, gb_routes_list)


    # ========== 2) 真实评价初始解 ==========
    init_eval, routes_clusters = evaluator.evaluate_init_saa(routes_clusters, scenarios)
    work = pack_solution(routes_clusters, init_eval)
    best = copy.deepcopy(work)
    print("first——————init_cost", work["cost"], work["ra"])
    


    # ========== 3) 准备 VNS: Shaker + VND ==========
    rng = np.random.default_rng(0)

    # distance matrix name depends on你的 problem 定义，做一个稳健 fallback
    D = getattr(problem, "total_distance_matrix", None)
    if D is None:
        D = getattr(problem, "distance_matrix", None)
    if D is None:
        raise AttributeError("problem 中找不到 total_distance_matrix / distance_matrix")

    # Neighborhood: 一次 move 的算子集合（按你当前版本：problem=...）
    nb = Neighborhood(
        problem=problem,
        removal_ratio=0.30,
        alpha_boundary=0.50,
        min_gb_len=2,
        rng=rng,
        evaluator=None,  # 这里不使用 _cost_increase 系列时可以 None
    )

    shaker = Shaker(nb, NS)
    vnd = VNDRefiner(
        nb,
        evaluator=lambda routes: evaluator.evaluate_saa(routes, scenarios),
        eval_proxy=lambda r: evaluator.evaluate_saa(r, scenarios[:5]),
        dominates_fn=dominates_cost,
        rng=rng,
        eps=1e-9
    )
    
    # ========== 4) VNS 主循环（只看 cost；早停） ==========
    eps = 1e-9
    max_iter = 2000
    patience = 100
    no_improve = 0
    k_max = len(NS)

    print(f"[VNS] start_cost = {best['cost']:.6f}")

    history = [(0, best["cost"], work["cost"])]

    for it in range(1, max_iter + 1):
        improved_this_iter = False

        # base = 当前工作解（本轮内会被接受更新）
        base = copy.deepcopy(work)

        k = 1
        while k <= k_max:
            base_routes = copy.deepcopy(base["routes_by_clusters"])

            # 1) shaking：必须从 base_routes 出发
            cand_routes = shaker.shake(base_routes, k=k, cid=None)

            # 2) VND refine（内部 cost-only 接受；这里外层也只看 cost）
            cand_routes, cand_cend, _ = vnd.refine(
                cand_routes,
                NL=NL,
                cid=None,
                tries_per_op=2,
                max_steps=30,
            )

             # 3) pack cand solution
            cand = pack_solution(cand_routes, cand_cend)

            # 4) 接受准则：相对 base（只看 cost）
            if cand["cost"] < base["cost"] - eps:
                base = cand
                k = 1
                improved_this_iter = True
            else:
                k += 1

        # 一轮结束：更新 work
        work = base

        # 更新 best
        if work["cost"] < best["cost"] - eps:
            best = copy.deepcopy(work)

        history.append((it, best["cost"], work["cost"]))

        # 早停
        if improved_this_iter:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[VNS] early stop: no improvement for {patience} cycles")
                break

        if it % 100 == 0:
            print(f"[VNS] it={it:4d}  best={best['cost']:.6f}  work={work['cost']:.6f}  no_improve={no_improve}")

    print("[VNS] final_best_cost =", best["cost"])

    end_time = time.time()
    running_time = end_time - start_time
    print("running_time", running_time)

    print("best total =", best["cend"]["total_cost"])
    print("best stage1 =", best["cend"]["stage1_cost"])
    print("best stage2 =", best["cend"]["stage2_cost"])
    print("check total - stage1 - stage2 =",
        best["cend"]["total_cost"] - best["cend"]["stage1_cost"] - best["cend"]["stage2_cost"])


    best_flat_routes = best["cend"]["decoded_routes"]
    print('best_flat_routes', best_flat_routes)
    plot_routes(problem, best_flat_routes)
            



    











# =========================
# 你自己的全局参数
# =========================
BETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
N_RUNS = 10


def run_once(instance_path: str, beta: float, run_id: int, seed: int):
    """
    单次实验：给定 beta + seed，跑完整流程并返回一个 dict（用于写 Excel）
    """
    t_stage2 = 0.0
    # -----------------------------
    # 1. Load instance（你的原代码）
    # -----------------------------
    instance_name = os.path.splitext(os.path.basename(instance_path))[0]

    ev_params = {"soc_max": 40.0, "vehicle_capacity": 650.0}
    problem = load_problem(instance_path, ev_params)
    evaluator = Evaluator(problem, ra_safe=0.1, ra_risk=0.7)

    start_time = time.time()

    # -----------------------------
    # 2. Scenarios（固定100场景，但每次run用不同seed更合理）
    # -----------------------------
    mu = np.asarray(problem.demand, dtype=float)
    scenarios = generate_normal_scenarios(
        mu=mu,
        beta=beta,
        n_scenarios=100,
        random_state=seed,   # ✅ 每次 run 改 seed
    )

    # -----------------------------
    # 3. clustering（可选：如果你希望每次run固定聚类结果，random_state固定即可）
    # -----------------------------
    clusters = clustering(
        problem,
        method="kmeans",
        feature="location",
        random_state=1,   # ✅ 固定聚类，避免实验噪声
    )

    

    depot_coords = np.asarray(
        [(problem.nodes[i].x, problem.nodes[i].y) for i in problem.depots],
        dtype=float
    )

    # -----------------------------
    # 4. 初始化：GB-ACO（你的原代码）
    # -----------------------------
    routes_clusters = {}
    for cid, cluster in clusters.items():
        customer_ids = list(cluster)
        points = np.asarray([(problem.nodes[i].x, problem.nodes[i].y) for i in customer_ids], dtype=float)

        gbs_info = gb_clustering(
            points,
            method="gb_kmeans",
            split_k=0.4,
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
    # 5. 初始解 SAA 评估
    # -----------------------------
    t_start_stage2 = time.time()
    init_eval, routes_clusters = evaluator.evaluate_init_saa(routes_clusters, scenarios)
    t_end_stage2 = time.time()
    t_stage2_1 = t_end_stage2 - t_start_stage2
    t_stage2 += t_stage2_1

    work = pack_solution(routes_clusters, init_eval)
    best = copy.deepcopy(work)

    # -----------------------------
    # 6. VNS + VND（你现在的版本：proxy 5 场景 + full 100）
    # -----------------------------
    rng = np.random.default_rng(seed)  # ✅ 让邻域扰动也随 run_id 变化

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
        evaluator=lambda routes: evaluator.evaluate_saa(routes, scenarios),      # full 100
        eval_proxy=lambda r: evaluator.evaluate_saa(r, scenarios[:5]),          # proxy 5
        dominates_fn=dominates_cost,
        rng=rng,
        eps=1e-9
    )

    eps = 1e-9
    max_iter = 2000
    patience = 100
    no_improve = 0
    k_max = len(NS)

    for it in range(1, max_iter + 1):
        improved_this_iter = False
        base = copy.deepcopy(work)

        k = 1
        while k <= k_max:
            base_routes = copy.deepcopy(base["routes_by_clusters"])
            cand_routes = shaker.shake(base_routes, k=k, cid=None)

            cand_routes, cand_cend, cand_info = vnd.refine(
                cand_routes,
                NL=NL,
                cid=None,
                tries_per_op=2,
                max_steps=30,
            )

            t_stage2 += cand_info["t_stage2"]

            cand = pack_solution(cand_routes, cand_cend)

            if cand["cost"] < base["cost"] - eps:
                base = cand
                k = 1
                improved_this_iter = True
            else:
                k += 1

        work = base
        if work["cost"] < best["cost"] - eps:
            best = copy.deepcopy(work)

        if improved_this_iter:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    runtime = time.time() - start_time
    t_stage1 = runtime - t_stage2

    # -----------------------------
    # 7. 返回记录（用于写Excel）
    # -----------------------------
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "instance": instance_name,
        "beta": beta,
        "run_id": run_id,
        "seed": seed,
        "init_cost": float(work["cost"]) if run_id == -1 else float(init_eval["total_cost"]),
        "best_cost": float(best["cost"]),
        "stage1_cost": float(best["cend"].get("stage1_cost", np.nan)),
        "stage2_cost": float(best["cend"].get("stage2_cost", np.nan)),
        "runtime_sec": float(runtime),
        "saa_S": int(scenarios.shape[0]),
        "proxy_S": 5,

        "t_stage1": t_stage1,
        "t_stage2": t_stage2,

    }




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

    # instance_name = 'c103_21'
    # base_path = r"D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)"
    # file_path = os.path.join(base_path, f"{instance_name}.txt")

    # path = r'D:\02_Research\DataSet\SPR'

    # run_experiments_resume_ultimate(
    #     instance_source=path,
    #     results_dir=r"D:\02_Research\Results"
    # )
    main()









# # if __name__ == "__main__":
#     main()


