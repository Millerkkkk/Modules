



import os
from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from evaluate import Evaluator
from main_CCP_SD_MOP import pack_solution


import time

from Clustering.api import clustering
from EVRP.loader import load_problem
from CustomerPlanning_GA import plan_internal_gbs_order_ga
from GB.api import gb_clustering
from GbPlanning_GA import plan_gb_order_ga
from params import GAParams, ev_params



from scipy.stats import norm
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import pickle


from comparison.run_pymoo.problem import (
    GBClusterRoutingProblemDual,
    GBHeuristicSamplingDual,
    MyOutput,
    build_block_state,
    decode_routes_clusters,
    decode_routes_clusters_dual,
)
from comparison.run_pymoo.problem import GBClusterRoutingProblem
from comparison.run_pymoo.problem import GBHeuristicSampling

from pymoo.termination.default import DefaultMultiObjectiveTermination
from stochastic_demand import fast_forward_selection, generate_normal_scenarios

from pymoo.termination import get_termination
from pymoo.optimize import minimize
from comparison.run_pymoo.problem import GBClusterRoutingProblem, GBHeuristicSampling, build_block_state, load_instance, load_instance_ffs, solve
from comparison.run_pymoo.run_CMOPSO import build_algorithm_cmopso
from comparison.run_pymoo.run_MOEAD import build_algorithm_moead
from comparison.run_pymoo.run_NSGA2 import build_algorithm_nsga2
from comparison.run_pymoo.run_NSGA3 import build_algorithm_nsga3





def run_once(
    instance_path,
    beta,
    alpha,
    ev_params, 
    algorithm,
    run_id,
    n_scenarios,
    n_reduced_scenarios,
    seed,

    pop_size = 100,
    N_PARTITIONS = 99,
    N_NEIGHBORS = 10,
    PROB_NEIGHBOR_MATING = 0.7,
):
    instance_name = os.path.splitext(os.path.basename(instance_path))[0]


    problem = load_problem(instance_path, ev_params)

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
    # 4. GB setup
    # =============================
    routes_clusters = {}
    gbs_customer_id_by_cluster = {}

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
        gbs_customer_id_by_cluster[cid] = gbs_customer_id

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
    

    evaluator = Evaluator(problem, ra_safe=0.1, ra_risk=0.7)

    # 单个高质量初始解
    init_eval, routes_clusters = evaluator.evaluate_init_ffs(
        routes_clusters, reduced_scenarios, reduced_probs
    )
    init_sol = pack_solution(routes_clusters, init_eval)

    # 随机 GB 初始种群
    population = generate_initial_population_gb(
        evaluator=evaluator,
        gbs_customer_id_by_cluster=gbs_customer_id_by_cluster,
        reduced_scenarios=reduced_scenarios,
        reduced_probs=reduced_probs,
        pop_size=pop_size,
        seed=seed,
        max_trials=pop_size * 30,
        verbose=1,
    )

    # 可选：把高质量单解替换掉种群里一个最差的，或者直接插进去
    population[0] = init_sol




    algo_builders = {
        "cmopso": build_algorithm_cmopso(pop_size),
        "moead": build_algorithm_moead(N_PARTITIONS, N_NEIGHBORS, PROB_NEIGHBOR_MATING),
        "nsga2": build_algorithm_nsga2(pop_size),
        "nsga3": build_algorithm_nsga3(N_PARTITIONS),
    }

    if algorithm not in algo_builders:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    alg = algo_builders[algorithm]

    begin_time = time.time()
    res = solve(problem, alg, seed=seed, verbose=True)
    runtime_sec = time.time() - begin_time

    # -------------------------
    # 解析 pymoo 返回结果
    # -------------------------
    F = res.F
    X = res.X

    # 防止单目标 / 单解 / 多解格式不一致
    if F is None:
        F_array = np.empty((0, 2))
    else:
        F_array = np.atleast_2d(F)

    archive_size = len(F_array)

    # 假设你的两个目标分别是 [cost, ra]
    pareto_points = []
    for row in F_array:
        if len(row) >= 2:
            pareto_points.append((float(row[0]), float(row[1])))
        elif len(row) == 1:
            pareto_points.append((float(row[0]), float("nan")))

    if archive_size > 0:
        cost_values = [p[0] for p in pareto_points]
        ra_values = [p[1] for p in pareto_points]

        min_cost_idx = int(np.argmin(cost_values))
        min_ra_idx = int(np.argmin(ra_values))

        best_cost_in_archive = float(cost_values[min_cost_idx])
        ra_of_best_cost = float(ra_values[min_cost_idx])

        best_ra_in_archive = float(ra_values[min_ra_idx])
        cost_of_best_ra = float(cost_values[min_ra_idx])

        # 这里“final”先定义成 archive 中 cost 最小的解
        final_cost = best_cost_in_archive
        final_ra = ra_of_best_cost
    else:
        best_cost_in_archive = float("nan")
        ra_of_best_cost = float("nan")
        best_ra_in_archive = float("nan")
        cost_of_best_ra = float("nan")
        final_cost = float("nan")
        final_ra = float("nan")

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "instance": instance_name,
        "beta": float(beta),
        "alpha": float(alpha),
        "algorithm": algorithm,
        "run_id": int(run_id),
        "seed": int(seed),
        "n_scenarios": int(n_scenarios),

        # 由于当前 res 里没有 init_sol，先留空
        "init_cost": float("nan"),
        "init_ra": float("nan"),

        "final_cost": float(final_cost),
        "final_ra": float(final_ra),

        "best_cost_in_archive": float(best_cost_in_archive),
        "ra_of_best_cost": float(ra_of_best_cost),

        "best_ra_in_archive": float(best_ra_in_archive),
        "cost_of_best_ra": float(cost_of_best_ra),

        "archive_size": int(archive_size),
        "n_candidates": int(archive_size),   # 暂时用 archive_size 代替
        "n_iterations": int(len(res.history)) if hasattr(res, "history") and res.history is not None else -1,
        "runtime_sec": float(runtime_sec),

        # 明细
        "pareto_points": pareto_points,
        "X": X.tolist() if X is not None and hasattr(X, "tolist") else X,
        "F": F.tolist() if F is not None and hasattr(F, "tolist") else F,
    }

    return result









def run_once_spr(
    instance_path: str,
    beta: float,
    alpha: float,
    ev_params,
    algorithm,
    run_id,
    n_scenarios: int = 100,
    n_reduced_scenarios: int = 10,
    base_seed: int = 42,
    pop_size: int = 50,
    n_seed_solutions: int = 10,
):
    problem = load_problem(instance_path, ev_params)
    instance_name = os.path.splitext(os.path.basename(instance_path))[0]

    seed = int(base_seed + run_id)

    mu = np.asarray(problem.demand, dtype=float)
    scenarios = generate_normal_scenarios(
        mu=mu,
        beta=beta,
        n_scenarios=n_scenarios,
        random_state=seed,
    )

    _, reduced_scenarios, reduced_probs = fast_forward_selection(
        scenarios=scenarios,
        n_select=n_reduced_scenarios,
    )

    evaluator_obj = Evaluator(problem, ra_safe=0.1, ra_risk=0.7)

    def evaluator_fn(routes_clusters):
        cend = evaluator_obj.evaluate_ffs(
            routes_clusters,
            reduced_scenarios,
            reduced_probs,
        )
        if isinstance(cend, tuple) and len(cend) == 2:
            return cend[0]
        return cend

    clusters, seed_solutions = generate_gb_seed_solutions(
        problem=problem,
        base_seed=seed,
        n_seed_solutions=n_seed_solutions,
    )

    base_routes_clusters = seed_solutions[0]
    # state = build_block_state(base_routes_clusters)

    # mo_problem = GBClusterRoutingProblem(
    #     state=state,
    #     evaluator_fn=evaluator_fn,
    # )

    # sampling = GBHeuristicSampling(
    #     state=state,
    #     seed_solutions=seed_solutions,
    # )
    
    state = build_block_state(base_routes_clusters)

    mo_problem = GBClusterRoutingProblemDual(
        state=state,
        evaluator_fn=evaluator_fn,
    )

    sampling = GBHeuristicSamplingDual(
        state=state,
        seed_solutions=seed_solutions,
    )


    if algorithm == "cmopso":
        alg = build_algorithm_cmopso(pop_size=pop_size, sampling=sampling)
    elif algorithm == "moead":
        alg = build_algorithm_moead(
            N_PARTITIONS, N_NEIGHBORS, PROB_NEIGHBOR_MATING, sampling=sampling
        )
    elif algorithm == "nsga2":
        alg = build_algorithm_nsga2(pop_size=pop_size, sampling=sampling)
    elif algorithm == "nsga3":
        alg = build_algorithm_nsga3(N_PARTITIONS, sampling=sampling)

    # termination = DefaultMultiObjectiveTermination(
    #     period=1000,
    #     n_max_gen=20000
    # )

    termination = get_termination("time", 50)   # 600秒

    start_time = time.time()

    res = minimize(
        mo_problem,
        alg,
        termination=termination,
        seed=seed,
        verbose=True,
        output=MyOutput()
    )

    runtime_sec = time.time() - start_time

    F = res.F
    X = res.X

    # if X is None:
    #     pareto_routes = []
    # elif np.ndim(X) == 1:
    #     pareto_routes = [decode_routes_clusters(X, state)]
    # else:
    #     pareto_routes = [decode_routes_clusters(x, state) for x in X]

    if X is None:
        pareto_routes = []
    elif np.ndim(X) == 1:
        pareto_routes = [decode_routes_clusters_dual(X, state)]
    else:
        pareto_routes = [decode_routes_clusters_dual(x, state) for x in X]


    if F is None:
        F_array = np.empty((0, 2))
    else:
        F_array = np.atleast_2d(F)

    archive_size = len(F_array)

    pareto_points = []
    for row in F_array:
        if len(row) >= 2:
            pareto_points.append((float(row[0]), float(row[1])))
        elif len(row) == 1:
            pareto_points.append((float(row[0]), float("nan")))

    if archive_size > 0:
        cost_values = [p[0] for p in pareto_points]
        ra_values = [p[1] for p in pareto_points]

        min_cost_idx = int(np.argmin(cost_values))
        min_ra_idx = int(np.argmin(ra_values))

        best_cost_in_archive = float(cost_values[min_cost_idx])
        ra_of_best_cost = float(ra_values[min_cost_idx])

        best_ra_in_archive = float(ra_values[min_ra_idx])
        cost_of_best_ra = float(cost_values[min_ra_idx])

        final_cost = best_cost_in_archive
        final_ra = ra_of_best_cost
    else:
        best_cost_in_archive = float("nan")
        ra_of_best_cost = float("nan")
        best_ra_in_archive = float("nan")
        cost_of_best_ra = float("nan")
        final_cost = float("nan")
        final_ra = float("nan")

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "instance": instance_name,
        "beta": float(beta),
        "alpha": float(alpha),
        "algorithm": algorithm,
        "run_id": int(run_id),
        "seed": int(seed),
        "n_scenarios": int(n_scenarios),

        "init_cost": float("nan"),
        "init_ra": float("nan"),

        "final_cost": float(final_cost),
        "final_ra": float(final_ra),

        "best_cost_in_archive": float(best_cost_in_archive),
        "ra_of_best_cost": float(ra_of_best_cost),

        "best_ra_in_archive": float(best_ra_in_archive),
        "cost_of_best_ra": float(cost_of_best_ra),

        "archive_size": int(archive_size),
        "n_candidates": int(archive_size),   # placeholder
        "n_iterations": int(len(res.history)) if hasattr(res, "history") and res.history is not None else -1,
        "runtime_sec": float(runtime_sec),

        "pareto_points": pareto_points,
        "X": X.tolist() if X is not None and hasattr(X, "tolist") else X,
        "F": F.tolist() if F is not None and hasattr(F, "tolist") else F,
    }

    return result







def _make_key(instance, beta, run_id):
    return f"{instance}|{float(beta):.6f}|{int(run_id)}"


def _list_instances(instance_source: str):
    if os.path.isfile(instance_source):
        return [instance_source]

    if os.path.isdir(instance_source):
        files = []
        for name in os.listdir(instance_source):
            full_path = os.path.join(instance_source, name)
            if os.path.isfile(full_path):
                files.append(full_path)
        files.sort()
        return files

    raise FileNotFoundError(f"instance_source not found: {instance_source}")


def _append_row_csv(csv_path, row_dict):
    df = pd.DataFrame([row_dict])
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False)


def _append_rows_csv(csv_path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False)


def _split_run_record(rec):
    summary_keys = [
        "timestamp",
        "instance",
        "beta",
        "alpha",
        "algorithm",
        "run_id",
        "seed",
        "n_scenarios",
        "init_cost",
        "init_ra",
        "final_cost",
        "final_ra",
        "best_cost_in_archive",
        "ra_of_best_cost",
        "best_ra_in_archive",
        "cost_of_best_ra",
        "archive_size",
        "n_candidates",
        "n_iterations",
        "runtime_sec",
    ]
    summary_rec = {k: rec.get(k, None) for k in summary_keys}
    detail_rec = {k: v for k, v in rec.items() if k not in summary_keys}
    return summary_rec, detail_rec


def _save_run_details_pkl(details_dir, instance_name, beta, run_id, detail_rec):
    os.makedirs(details_dir, exist_ok=True)
    file_name = f"{instance_name}_beta_{beta}_run_{run_id}.pkl"
    path = os.path.join(details_dir, file_name)
    with open(path, "wb") as f:
        pickle.dump(detail_rec, f)
    return path


def _make_archive_rows(rec):
    rows = []
    pareto_points = rec.get("pareto_points", [])
    for i, point in enumerate(pareto_points, start=1):
        cost = point[0] if len(point) > 0 else None
        ra = point[1] if len(point) > 1 else None
        rows.append({
            "instance": rec.get("instance"),
            "beta": rec.get("beta"),
            "run_id": rec.get("run_id"),
            "seed": rec.get("seed"),
            "point_id": i,
            "cost": cost,
            "ra": ra,
        })
    return rows


def _build_summary(df_ok: pd.DataFrame):
    if df_ok.empty:
        return pd.DataFrame()

    agg_dict = {
        "final_cost": ["mean", "std", "min", "max"],
        "final_ra": ["mean", "std", "min", "max"],
        "best_cost_in_archive": ["mean", "std", "min", "max"],
        "best_ra_in_archive": ["mean", "std", "min", "max"],
        "archive_size": ["mean", "std", "min", "max"],
        "runtime_sec": ["mean", "std", "min", "max"],
    }

    group_cols = ["instance", "beta"]
    summary = df_ok.groupby(group_cols).agg(agg_dict)
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    summary["n_runs"] = df_ok.groupby(group_cols).size().values
    return summary


def _run_one_task(args):
    instance_path, instance_name, beta, alpha, ev_params, algorithm, n_scenarios, run_id, seed = args

    try:
        rec = run_once(
            instance_path=instance_path,
            beta=beta,
            alpha=alpha,
            ev_params=ev_params,
            algorithm=algorithm,
            run_id=run_id,
            n_scenarios=n_scenarios,
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
    ev_params: dict,
    results_dir: Optional[str] = None,
    base_seed: int = 42,
    alpha: float = 0.0,
    algorithm: str = "nsga2",
    n_scenarios: int = 1,
    update_summary_each_run: bool = True,
    continue_on_error: bool = True,
    max_workers: int = 10,
):
    """
    支持：
    - 多实例
    - 多 beta
    - 每组重复 N_RUNS 次
    - 断点续跑
    - 每个 run 写一行 progress_runs.csv
    - Pareto 点写入 pareto_archive.csv
    - 完整详情写入 run_details/*.pkl
    - 最终导出 final_results.xlsx
    """

    # -------------------------
    # 创建输出目录
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
        required_cols = {"instance", "beta", "run_id"}
        if required_cols.issubset(df_done.columns):
            for _, row in df_done.iterrows():
                done.add(_make_key(row["instance"], row["beta"], row["run_id"]))
        print(f"[RESUME] Loaded progress: {progress_csv} | completed={len(done)}")
    else:
        print(f"[RESUME] No progress file. Will create: {progress_csv}")

    # -------------------------
    # 1) 构造任务
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

                seed = base_seed + 1000 * int(round(beta * 100)) + run_id

                tasks.append((
                    instance_path,
                    instance_name,
                    beta,
                    alpha,
                    ev_params,
                    algorithm,
                    n_scenarios,
                    run_id,
                    seed,
                ))

    print(f"[TASKS] total={total_tasks} | skipped(done)={skipped} | remaining={len(tasks)}")

    # -------------------------
    # 2) 没有剩余任务时也导出最终结果
    # -------------------------
    if len(tasks) == 0:
        if os.path.exists(progress_csv):
            df_all = pd.read_csv(progress_csv)
            if "status" in df_all.columns:
                df_ok = df_all[df_all["status"] == "completed"].copy()
            else:
                df_ok = df_all.copy()

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
        else:
            print("[DONE] No task and no progress file found.")
        return

    # -------------------------
    # 3) 并行运行
    # -------------------------
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_run_one_task, task): task
            for task in tasks
        }

        for idx, future in enumerate(as_completed(future_to_task), 1):
            instance_path, instance_name, beta, alpha, ev_params, algorithm, n_scenarios, run_id, seed = future_to_task[future]
            print(
                f"\n=== [{idx}/{len(tasks)} DONE] "
                f"instance={instance_name} beta={beta:.3f} run={run_id}/{N_RUNS} seed={seed} ==="
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
                    if "status" in df_all.columns:
                        df_ok = df_all[df_all["status"] == "completed"].copy()
                    else:
                        df_ok = df_all.copy()

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
    # 4) 最终导出 Excel
    # -------------------------
    if os.path.exists(progress_csv):
        df_all = pd.read_csv(progress_csv)

        if "status" in df_all.columns:
            df_ok = df_all[df_all["status"] == "completed"].copy()
        else:
            df_ok = df_all.copy()

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

    INSTANCE_NAME = "rc103_21"
    BASE_PATH = r"D:\02_Research\DataSet\evrptw_instances_LijunFan\large_instances(100customer21cs_10)\50"
    file_path = os.path.join(BASE_PATH, f"{INSTANCE_NAME}.txt")

    SEED = 1
    POP_SIZE = 100

    # =========================
    # 你自己的全局参数
    # =========================
    BETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    N_RUNS = 10

    ## ======== 整个文件夹运行
    path = r'D:\02_Research\DataSet\SPR'

    ev_params = {
        "soc_max": 40.0,
        "vehicle_capacity": 650.0,
    }

    pop_size = 100 
    N_PARTITIONS = 99
    N_NEIGHBORS = 10
    PROB_NEIGHBOR_MATING = 0.7


    # run_once(file_path,0.1,0.9,ev_params,"cmopso",1,100,10, 0)

    run_once_spr(file_path,0.2,0.9,ev_params,"cmopso",1,100,1, 0)


    # run_experiments_resume_ultimate_10_worker(
    #     BETAS=BETAS,
    #     N_RUNS=N_RUNS,
    #     instance_source=path,
    #     ev_params=ev_params,
    #     results_dir=r"D:\02_Research\Results",
    #     base_seed=42,
    #     alpha=0.9,
    #     algorithm="nsga2",
    #     n_scenarios=20,
    #     update_summary_each_run=True,
    #     continue_on_error=True,
    #     max_workers=10,
    # )


    # run_experiments_resume_ultimate_10_worker(
    #     BETAS=BETAS,
    #     N_RUNS=N_RUNS,
    #     instance_source=path,
    #     ev_params=ev_params,
    #     results_dir=r"D:\02_Research\Results",
    #     base_seed=42,
    #     alpha=0.9,
    #     algorithm="nsga3",
    #     n_scenarios=20,
    #     update_summary_each_run=True,
    #     continue_on_error=True,
    #     max_workers=10,
    # )


    # run_experiments_resume_ultimate_10_worker(
    #     BETAS=BETAS,
    #     N_RUNS=N_RUNS,
    #     instance_source=path,
    #     ev_params=ev_params,
    #     results_dir=r"D:\02_Research\Results",
    #     base_seed=42,
    #     alpha=0.9,
    #     algorithm="cmopso",
    #     n_scenarios=20,
    #     update_summary_each_run=True,
    #     continue_on_error=True,
    #     max_workers=10,
    # )


    # run_experiments_resume_ultimate_10_worker(
    #     BETAS=BETAS,
    #     N_RUNS=N_RUNS,
    #     instance_source=path,
    #     ev_params=ev_params,
    #     results_dir=r"D:\02_Research\Results",
    #     base_seed=42,
    #     alpha=0.9,
    #     algorithm="moead",
    #     n_scenarios=20,
    #     update_summary_each_run=True,
    #     continue_on_error=True,
    #     max_workers=10,
    # )




