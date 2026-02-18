from dataclasses import dataclass
import os
import random
import sys
import time
from typing import Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "./.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)



from params import ACOParams
import copy

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


import copy
from collections import Counter

def pack_solution(routes_by_clusters, cend):
    return {
        "routes_by_clusters": copy.deepcopy(routes_by_clusters),
        "cost": float(cend["total_cost"]),
        "ra": float(cend.get("total_ra", 0.0)),
        "cend": cend,
    }

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
    instance_name = 'c103_21'
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
            split_k=0.4,
            merge=True,
            random_state=None,
        )

        gbs_customer_id = [[customer_ids[j] for j in cl] for cl in gbs_info.gbs]
        gbs_by_cluster[cid] = gbs_customer_id

        # ===== 规划 gbs 之间的路径 =====
        params_GB = ACOParams(num_ants=10, num_iter=60)
        gb_order_result = plan_gb_order(gbs_info.centers, depot_coords, params_GB)

        gb_order_by_cluster[cid] = gb_order_result.order

        # ===== 规划 gb 内客户的路径  =====
        params_customer = ACOParams(num_ants=20, num_iter=100, seed=0)
        routes = plan_internal_gbs_order(
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
    init_eval, routes_clusters = evaluator.evaluate_init(routes_clusters)
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
    vnd = VNDRefiner(nb, evaluator=evaluator.evaluate_clusters, rng=rng, eps=1e-9)

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


    best_flat_routes = best["cend"]["decoded_routes"]
    print('best_flat_routes', best_flat_routes)
    plot_routes(problem, best_flat_routes)
            



    






if __name__ == "__main__":
    main()


