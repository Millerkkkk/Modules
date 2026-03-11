import os
import pickle
from datetime import datetime
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from main_CCP_SD_MOP import run_once_ccp
from main_SPR_SD_MOP import run_once_spr

from Experiment.runner import run_experiments_parallel_spr, run_experiments_parallel_ccp


# # =========================
# # 工具函数：列举 instances
# # =========================
# def _list_instances(instance_source: str):
#     if os.path.isfile(instance_source):
#         return [instance_source]

#     return sorted(
#         os.path.join(instance_source, f)
#         for f in os.listdir(instance_source)
#         if f.lower().endswith(".txt")
#     )


# # =========================
# # 工具函数：任务唯一 key
# # =========================
# def _make_key(instance: str, beta: float, alpha: float, run_id: int):
#     return (
#         str(instance),
#         round(float(beta), 10),
#         round(float(alpha), 10),
#         int(run_id),
#     )


# # =========================
# # 工具函数：安全追加一行 CSV
# # =========================
# def _append_row_csv(path: str, record: dict):
#     df_row = pd.DataFrame([record])
#     write_header = not os.path.exists(path)
#     df_row.to_csv(path, mode="a", header=write_header, index=False)



# # =========================
# # 工具函数：安全追加多行 CSV
# # =========================
# def _append_rows_csv(path: str, rows: list[dict]):
#     if not rows:
#         return
#     df_rows = pd.DataFrame(rows)
#     write_header = not os.path.exists(path)
#     df_rows.to_csv(path, mode="a", header=write_header, index=False)



# # =========================
# # 工具函数：从 progress 构建 summary
# # =========================
# def _build_summary(df: pd.DataFrame):
#     if df.empty:
#         return pd.DataFrame()

#     agg_dict = {
#         "final_cost": ["count", "mean", "std", "min"],
#         "final_ra": ["mean", "std", "min"],
#         "best_cost_in_archive": ["mean", "std", "min"],
#         "best_ra_in_archive": ["mean", "std", "min"],
#         "archive_size": ["mean", "std", "max"],
#         "runtime_sec": ["mean", "std"],
#         "n_candidates": ["mean", "std"],
#         "n_iterations": ["mean", "std"],
#     }

#     if "t_stage1" in df.columns:
#         agg_dict["t_stage1"] = ["mean", "std"]
#     if "t_stage2" in df.columns:
#         agg_dict["t_stage2"] = ["mean", "std"]
    
#     if "cost_stage1" in df.columns:
#         agg_dict["cost_stage1"] = ["mean", "std"]
#     if "cost_stage2" in df.columns:
#         agg_dict["cost_stage2"] = ["mean", "std"]

#     if "ra_stage1" in df.columns:
#         agg_dict["ra_stage1"] = ["mean", "std"]
#     if "ra_stage2" in df.columns:
#         agg_dict["ra_stage2"] = ["mean", "std"]

#     summary = (
#         df.groupby(["instance", "beta", "alpha"])
#           .agg(agg_dict)
#     )

#     summary.columns = ["_".join(col).strip() for col in summary.columns.to_flat_index()]
#     summary = summary.reset_index()

#     if "t_stage2_mean" in summary.columns and "runtime_sec_mean" in summary.columns:
#         summary["stage2_ratio_mean"] = summary["t_stage2_mean"] / summary["runtime_sec_mean"]

#     return summary


# # =========================
# # 工具函数：保存详细对象为 PKL
# # =========================
# def _save_run_details_pkl(
#     details_dir: str,
#     instance_name: str,
#     beta: float,
#     alpha: float,
#     run_id: int,
#     detail_rec: dict,
# ):
#     os.makedirs(details_dir, exist_ok=True)
#     fname = f"{instance_name}_beta{beta:.6f}_alpha{alpha:.6f}_run{run_id}.pkl"
#     path = os.path.join(details_dir, fname)
#     with open(path, "wb") as f:
#         pickle.dump(detail_rec, f)
#     return path


# # =========================
# # 工具函数：判断两个解是否同一点
# # =========================
# def same_point(a, b, eps=1e-9):
#     if a is None or b is None:
#         return False
#     return (
#         abs(float(a["cost"]) - float(b["cost"])) <= eps
#         and abs(float(a["ra"]) - float(b["ra"])) <= eps
#     )


# # =========================
# # 工具函数：把 archive 摊平成多行
# # =========================
# def _make_archive_rows(rec: dict):
#     archive = rec.get("archive", [])
#     final_best = rec.get("final_best", None)
#     min_cost_sol = rec.get("min_cost_sol", None)
#     min_ra_sol = rec.get("min_ra_sol", None)

#     rows = []
#     for i, s in enumerate(archive):
#         rows.append({
#             "timestamp": rec.get("timestamp"),
#             "instance": rec.get("instance"),
#             "beta": float(rec.get("beta")),
#             "alpha": float(rec.get("alpha")),
#             "run_id": int(rec.get("run_id")),
#             "seed": int(rec.get("seed")),
#             "sol_id": int(i),
#             "cost": float(s["cost"]),
#             "ra": float(s["ra"]),
#             "is_final_best": bool(same_point(s, final_best)) if final_best is not None else False,
#             "is_min_cost": bool(same_point(s, min_cost_sol)) if min_cost_sol is not None else False,
#             "is_min_ra": bool(same_point(s, min_ra_sol)) if min_ra_sol is not None else False,
#         })
#     return rows


# # =========================
# # 工具函数：拆分 summary / detail
# # =========================
# def _split_run_record(rec: dict):
#     detail_keys = {
#         "archive",
#         "work_history",
#         "final_best",
#         "min_cost_sol",
#         "min_ra_sol",
#         "problem",
#         "scenarios",
#         "cand_points",
#         "debug_log",
#         "clusters",
#     }

#     detail_rec = {k: rec[k] for k in rec if k in detail_keys}
#     summary_rec = {k: rec[k] for k in rec if k not in detail_keys}
#     return summary_rec, detail_rec


# # =========================
# # 终极稳定版：多实例 + BETAS x ALPHAS x N_RUNS + 断点续跑
# # =========================
# def run_experiments(
#     BETAS,
#     ALPHAS,
#     N_RUNS,
#     instance_source: str,
#     results_dir: Optional[str] = None,
#     base_seed: int = 42,
#     update_summary_each_run: bool = True,
#     continue_on_error: bool = True,
# ):
#     """
#     支持：
#     - 多实例（单文件或文件夹）
#     - BETAS x ALPHAS x N_RUNS 全组合
#     - 断点续跑
#     - 每次 run 立即落盘
#     - 保存 Pareto archive 和详细对象

#     输出：
#     - progress_runs.csv     : 每个 run 的摘要，一行一条
#     - summary_live.csv      : 实时聚合 summary
#     - failed_runs.csv       : 失败记录
#     - pareto_archive.csv    : 每个 Pareto 点一行
#     - run_details/*.pkl     : 每个 run 的详细对象
#     - final_results.xlsx    : 最终导出 runs + summary + pareto_archive
#     """

#     # -------------------------
#     # 0) 基本检查
#     # -------------------------
#     if not os.path.exists(instance_source):
#         raise FileNotFoundError(f"instance_source does not exist: {instance_source}")

#     instance_list = _list_instances(instance_source)
#     if len(instance_list) == 0:
#         raise ValueError(f"No .txt instances found in: {instance_source}")

#     # -------------------------
#     # 1) 创建输出目录
#     # -------------------------
#     run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

#     if results_dir is None:
#         results_dir = os.path.join(os.getcwd(), "beta_alpha_results", run_tag)
#     else:
#         results_dir = os.path.join(results_dir, run_tag)

#     os.makedirs(results_dir, exist_ok=True)

#     progress_csv = os.path.join(results_dir, "progress_runs.csv")
#     summary_live_csv = os.path.join(results_dir, "summary_live.csv")
#     failed_csv = os.path.join(results_dir, "failed_runs.csv")
#     pareto_csv = os.path.join(results_dir, "pareto_archive.csv")
#     final_xlsx = os.path.join(results_dir, "final_results.xlsx")
#     details_dir = os.path.join(results_dir, "run_details")

#     # -------------------------
#     # 2) 读取已完成记录，支持断点续跑
#     # -------------------------
#     done = set()
#     if os.path.exists(progress_csv):
#         df_done = pd.read_csv(progress_csv)
#         needed = {"instance", "beta", "alpha", "run_id"}
#         if needed.issubset(df_done.columns):
#             for _, row in df_done.iterrows():
#                 if "status" not in df_done.columns or row.get("status", "completed") == "completed":
#                     done.add(_make_key(row["instance"], row["beta"], row["alpha"], row["run_id"]))
#         print(f"[RESUME] Loaded progress: {progress_csv} | completed={len(done)}")
#     else:
#         print(f"[RESUME] No progress file. Will create: {progress_csv}")

#     # -------------------------
#     # 3) 生成任务列表
#     # -------------------------
#     tasks = []
#     total_tasks = 0
#     skipped = 0

#     for instance_path in instance_list:
#         instance_name = os.path.splitext(os.path.basename(instance_path))[0]

#         for beta in BETAS:
#             for alpha in ALPHAS:
#                 for run_id in range(1, N_RUNS + 1):
#                     total_tasks += 1
#                     key = _make_key(instance_name, beta, alpha, run_id)

#                     if key in done:
#                         skipped += 1
#                         continue

#                     # 可复现 seed 规则：instance + beta + alpha + run_id
#                     seed = (
#                         int(base_seed)
#                         + 1000000 * abs(hash(instance_name)) % 100000
#                         + 10000 * int(round(float(beta) * 1000))
#                         + 100 * int(round(float(alpha) * 1000))
#                         + int(run_id - 1)
#                     )

#                     tasks.append(
#                         (instance_path, instance_name, float(beta), float(alpha), int(run_id), int(seed))
#                     )

#     print(f"[TASKS] total={total_tasks} | skipped(done)={skipped} | remaining={len(tasks)}")

#     # -------------------------
#     # 4) 如果没有剩余任务，也刷新最终输出
#     # -------------------------
#     if len(tasks) == 0 and os.path.exists(progress_csv):
#         df_all = pd.read_csv(progress_csv)
#         df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
#         summary = _build_summary(df_ok)

#         if os.path.exists(pareto_csv):
#             df_pareto = pd.read_csv(pareto_csv)
#         else:
#             df_pareto = pd.DataFrame()

#         with pd.ExcelWriter(final_xlsx, engine="openpyxl") as writer:
#             df_all.to_excel(writer, sheet_name="runs", index=False)
#             summary.to_excel(writer, sheet_name="summary", index=False)
#             df_pareto.to_excel(writer, sheet_name="pareto_archive", index=False)

#         print(f"[DONE] Nothing to run. Final Excel refreshed: {final_xlsx}")
#         return {
#             "progress_csv": progress_csv,
#             "summary_live_csv": summary_live_csv,
#             "failed_csv": failed_csv,
#             "pareto_csv": pareto_csv,
#             "details_dir": details_dir,
#             "final_xlsx": final_xlsx,
#             "n_total": total_tasks,
#             "n_skipped": skipped,
#             "n_remaining": 0,
#         }

#     # -------------------------
#     # 5) 执行任务：每次 run 结束立即落盘
#     # -------------------------
#     for idx, (instance_path, instance_name, beta, alpha, run_id, seed) in enumerate(tasks, 1):
#         print(
#             f"\n=== [{idx}/{len(tasks)}] "
#             f"instance={instance_name} beta={beta:.6f} alpha={alpha:.6f} "
#             f"run={run_id}/{N_RUNS} seed={seed} ==="
#         )

#         try:
#             rec = run_once_ccp(
#                 instance_path=instance_path,
#                 beta=beta,
#                 alpha=alpha,
#                 run_id=run_id,
#                 seed=seed,
#             )

#             # 拆分摘要和详细对象
#             summary_rec, detail_rec = _split_run_record(rec)

#             # 保存详细对象
#             detail_path = _save_run_details_pkl(
#                 details_dir=details_dir,
#                 instance_name=instance_name,
#                 beta=beta,
#                 alpha=alpha,
#                 run_id=run_id,
#                 detail_rec=detail_rec,
#             )

#             # 补齐摘要关键字段
#             summary_rec["timestamp"] = summary_rec.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#             summary_rec["instance"] = summary_rec.get("instance", instance_name)
#             summary_rec["beta"] = float(summary_rec.get("beta", beta))
#             summary_rec["alpha"] = float(summary_rec.get("alpha", alpha))
#             summary_rec["run_id"] = int(summary_rec.get("run_id", run_id))
#             summary_rec["seed"] = int(summary_rec.get("seed", seed))
#             summary_rec["status"] = "completed"
#             summary_rec["details_path"] = detail_path

#             # 保存 progress
#             _append_row_csv(progress_csv, summary_rec)

#             # 保存 Pareto 点
#             archive_rows = _make_archive_rows(rec)
#             _append_rows_csv(pareto_csv, archive_rows)

#             # 更新 done
#             done.add(_make_key(summary_rec["instance"], summary_rec["beta"], summary_rec["alpha"], summary_rec["run_id"]))

#             # 实时更新 summary
#             if update_summary_each_run:
#                 df_all = pd.read_csv(progress_csv)
#                 df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
#                 summary = _build_summary(df_ok)
#                 summary.to_csv(summary_live_csv, index=False)

#         except Exception as e:
#             err = repr(e)
#             print(
#                 f"[ERROR] instance={instance_name} beta={beta} alpha={alpha} "
#                 f"run={run_id} failed: {err}"
#             )

#             fail_rec = {
#                 "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 "instance": instance_name,
#                 "beta": float(beta),
#                 "alpha": float(alpha),
#                 "run_id": int(run_id),
#                 "seed": int(seed),
#                 "status": "failed",
#                 "error": err,
#             }
#             _append_row_csv(failed_csv, fail_rec)

#             if not continue_on_error:
#                 raise

#             continue

#     # -------------------------
#     # 6) 跑完后生成最终 Excel
#     # -------------------------
#     if os.path.exists(progress_csv):
#         df_all = pd.read_csv(progress_csv)
#         df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
#         summary = _build_summary(df_ok)

#         if os.path.exists(pareto_csv):
#             df_pareto = pd.read_csv(pareto_csv)
#         else:
#             df_pareto = pd.DataFrame()

#         with pd.ExcelWriter(final_xlsx, engine="openpyxl") as writer:
#             df_all.to_excel(writer, sheet_name="runs", index=False)
#             summary.to_excel(writer, sheet_name="summary", index=False)
#             df_pareto.to_excel(writer, sheet_name="pareto_archive", index=False)

#         print(f"\n[DONE] Progress CSV : {progress_csv}")
#         if update_summary_each_run:
#             print(f"[DONE] Live Summary : {summary_live_csv}")
#         if os.path.exists(failed_csv):
#             print(f"[DONE] Failed Runs  : {failed_csv}")
#         if os.path.exists(pareto_csv):
#             print(f"[DONE] Pareto CSV   : {pareto_csv}")
#         print(f"[DONE] Details Dir   : {details_dir}")
#         print(f"[DONE] Final Excel   : {final_xlsx}")

#     else:
#         print("[DONE] No progress file found; nothing to export.")

#     return {
#         "progress_csv": progress_csv,
#         "summary_live_csv": summary_live_csv,
#         "failed_csv": failed_csv,
#         "pareto_csv": pareto_csv,
#         "details_dir": details_dir,
#         "final_xlsx": final_xlsx,
#         "n_total": total_tasks,
#         "n_skipped": skipped,
#         "n_remaining": len(tasks),
#     }





# # =========================
# # worker: 单任务执行
# # =========================
# def _run_one_task_ccp(args):
#     instance_path, instance_name, beta, alpha, run_id, seed = args
#     try:
#         rec = run_once_ccp(
#             instance_path=instance_path,
#             beta=beta,
#             alpha=alpha,
#             run_id=run_id,
#             seed=seed,
#         )
#         return {
#             "ok": True,
#             "instance_name": instance_name,
#             "beta": beta,
#             "alpha": alpha,
#             "run_id": run_id,
#             "seed": seed,
#             "record": rec,
#             "error": None,
#         }
#     except Exception as e:
#         return {
#             "ok": False,
#             "instance_name": instance_name,
#             "beta": beta,
#             "alpha": alpha,
#             "run_id": run_id,
#             "seed": seed,
#             "record": None,
#             "error": repr(e),
#         }


# # =========================
# # 完整可替换版：并行 + 断点续跑 + save_every + 内存缓存
# # =========================
# def run_experiments_parallel_ccp(
#     BETAS,
#     ALPHAS,
#     N_RUNS,
#     instance_source: str,
#     results_dir: Optional[str] = None,
#     base_seed: int = 42,
#     max_workers: int = 10,
#     update_summary_each_run: bool = True,
#     save_every: int = 10,
#     continue_on_error: bool = True,
#     create_timestamp_subdir: bool = False,
# ):
#     """
#     并行实验总控：
#     - 多实例
#     - BETAS x ALPHAS x N_RUNS
#     - 断点续跑
#     - 主进程安全写盘
#     - save_every 批量刷新 summary_live.csv
#     - progress_records 内存缓存，减少 read_csv(progress_csv) IO
#     """

#     if not os.path.exists(instance_source):
#         raise FileNotFoundError(f"instance_source does not exist: {instance_source}")

#     instance_list = _list_instances(instance_source)
#     if len(instance_list) == 0:
#         raise ValueError(f"No .txt instances found in: {instance_source}")

#     if results_dir is None:
#         results_dir = os.path.join(os.getcwd(), "parallel_results")

#     if create_timestamp_subdir:
#         run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
#         results_dir = os.path.join(results_dir, run_tag)

#     os.makedirs(results_dir, exist_ok=True)

#     progress_csv = os.path.join(results_dir, "progress_runs.csv")
#     summary_live_csv = os.path.join(results_dir, "summary_live.csv")
#     failed_csv = os.path.join(results_dir, "failed_runs.csv")
#     pareto_csv = os.path.join(results_dir, "pareto_archive.csv")
#     final_xlsx = os.path.join(results_dir, "final_results.xlsx")
#     details_dir = os.path.join(results_dir, "run_details")

#     # -------------------------
#     # 0) 读取已完成记录 + 内存缓存 progress
#     # -------------------------
#     done = set()
#     progress_records = []

#     if os.path.exists(progress_csv):
#         df_done = pd.read_csv(progress_csv)
#         progress_records = df_done.to_dict("records")

#         needed = {"instance", "beta", "alpha", "run_id"}
#         if needed.issubset(df_done.columns):
#             for _, row in df_done.iterrows():
#                 if "status" not in df_done.columns or row.get("status", "completed") == "completed":
#                     done.add(_make_key(row["instance"], row["beta"], row["alpha"], row["run_id"]))

#         print(f"[RESUME] Loaded progress: {progress_csv} | completed={len(done)}")
#     else:
#         print(f"[RESUME] No progress file. Will create: {progress_csv}")

#     # -------------------------
#     # 1) 生成任务
#     # -------------------------
#     tasks = []
#     total_tasks = 0
#     skipped = 0

#     for instance_path in instance_list:
#         instance_name = os.path.splitext(os.path.basename(instance_path))[0]

#         for beta in BETAS:
#             for alpha in ALPHAS:
#                 for run_id in range(1, N_RUNS + 1):
#                     total_tasks += 1
#                     key = _make_key(instance_name, beta, alpha, run_id)

#                     if key in done:
#                         skipped += 1
#                         continue

#                     # 稳定 seed：不要用 hash(instance_name)
#                     name_code = sum(ord(c) for c in instance_name)

#                     seed = (
#                         int(base_seed)
#                         + 100000 * name_code
#                         + 1000 * int(round(float(beta) * 100))
#                         + 10 * int(round(float(alpha) * 100))
#                         + int(run_id - 1)
#                     )

#                     tasks.append((
#                         instance_path,
#                         instance_name,
#                         float(beta),
#                         float(alpha),
#                         int(run_id),
#                         int(seed),
#                     ))

#     print(f"[TASKS] total={total_tasks} | skipped(done)={skipped} | remaining={len(tasks)}")

#     # -------------------------
#     # 2) 如果没有剩余任务，直接刷新最终输出
#     # -------------------------
#     if len(tasks) == 0:
#         df_all = pd.DataFrame(progress_records)
#         if df_all.empty:
#             print("[DONE] Nothing to run, and no progress found.")
#             return {
#                 "progress_csv": progress_csv,
#                 "summary_live_csv": summary_live_csv,
#                 "failed_csv": failed_csv,
#                 "pareto_csv": pareto_csv,
#                 "details_dir": details_dir,
#                 "final_xlsx": final_xlsx,
#                 "n_total": total_tasks,
#                 "n_skipped": skipped,
#                 "n_remaining": 0,
#             }

#         df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
#         summary = _build_summary(df_ok)
#         df_pareto = pd.read_csv(pareto_csv) if os.path.exists(pareto_csv) else pd.DataFrame()

#         with pd.ExcelWriter(final_xlsx, engine="openpyxl") as writer:
#             df_all.to_excel(writer, sheet_name="runs", index=False)
#             summary.to_excel(writer, sheet_name="summary", index=False)
#             df_pareto.to_excel(writer, sheet_name="pareto_archive", index=False)

#         if update_summary_each_run:
#             summary.to_csv(summary_live_csv, index=False)

#         print(f"[DONE] Nothing to run. Final Excel refreshed: {final_xlsx}")
#         return {
#             "progress_csv": progress_csv,
#             "summary_live_csv": summary_live_csv,
#             "failed_csv": failed_csv,
#             "pareto_csv": pareto_csv,
#             "details_dir": details_dir,
#             "final_xlsx": final_xlsx,
#             "n_total": total_tasks,
#             "n_skipped": skipped,
#             "n_remaining": 0,
#         }

#     # save_every 防御
#     if save_every is None or int(save_every) <= 0:
#         save_every = 1
#     save_every = int(save_every)

#     completed_since_last_summary = 0

#     # -------------------------
#     # 3) 并行执行
#     # -------------------------
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         future_to_task = {
#             executor.submit(_run_one_task, task): task
#             for task in tasks
#         }

#         for idx, future in enumerate(as_completed(future_to_task), 1):
#             instance_path, instance_name, beta, alpha, run_id, seed = future_to_task[future]

#             print(
#                 f"\n=== [{idx}/{len(tasks)} DONE] "
#                 f"instance={instance_name} beta={beta:.6f} alpha={alpha:.6f} "
#                 f"run={run_id}/{N_RUNS} seed={seed} ==="
#             )

#             try:
#                 out = future.result()
#             except Exception as e:
#                 err = repr(e)
#                 print(
#                     f"[ERROR] instance={instance_name} beta={beta} alpha={alpha} "
#                     f"run={run_id} failed: {err}"
#                 )

#                 fail_rec = {
#                     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                     "instance": instance_name,
#                     "beta": float(beta),
#                     "alpha": float(alpha),
#                     "run_id": int(run_id),
#                     "seed": int(seed),
#                     "status": "failed",
#                     "error": err,
#                 }
#                 _append_row_csv(failed_csv, fail_rec)

#                 if not continue_on_error:
#                     raise
#                 continue

#             if out["ok"]:
#                 rec = out["record"]

#                 summary_rec, detail_rec = _split_run_record(rec)

#                 detail_path = _save_run_details_pkl(
#                     details_dir=details_dir,
#                     instance_name=instance_name,
#                     beta=beta,
#                     alpha=alpha,
#                     run_id=run_id,
#                     detail_rec=detail_rec,
#                 )

#                 summary_rec["timestamp"] = summary_rec.get(
#                     "timestamp",
#                     datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 )
#                 summary_rec["instance"] = summary_rec.get("instance", instance_name)
#                 summary_rec["beta"] = float(summary_rec.get("beta", beta))
#                 summary_rec["alpha"] = float(summary_rec.get("alpha", alpha))
#                 summary_rec["run_id"] = int(summary_rec.get("run_id", run_id))
#                 summary_rec["seed"] = int(summary_rec.get("seed", seed))
#                 summary_rec["status"] = "completed"
#                 summary_rec["details_path"] = detail_path

#                 # 1) 立即落盘 progress
#                 _append_row_csv(progress_csv, summary_rec)

#                 # 2) 更新内存缓存
#                 progress_records.append(summary_rec)

#                 # 3) 立即落盘 pareto
#                 archive_rows = _make_archive_rows(rec)
#                 _append_rows_csv(pareto_csv, archive_rows)

#                 # 4) 更新 done
#                 done.add(_make_key(
#                     summary_rec["instance"],
#                     summary_rec["beta"],
#                     summary_rec["alpha"],
#                     summary_rec["run_id"]
#                 ))

#                 # 5) 批量刷新 summary_live
#                 if update_summary_each_run:
#                     completed_since_last_summary += 1

#                     if completed_since_last_summary >= save_every:
#                         df_all = pd.DataFrame(progress_records)
#                         df_ok = (
#                             df_all[df_all["status"] == "completed"].copy()
#                             if "status" in df_all.columns else df_all.copy()
#                         )
#                         summary = _build_summary(df_ok)
#                         summary.to_csv(summary_live_csv, index=False)
#                         completed_since_last_summary = 0

#             else:
#                 err = out["error"]
#                 print(
#                     f"[ERROR] instance={instance_name} beta={beta} alpha={alpha} "
#                     f"run={run_id} failed: {err}"
#                 )

#                 fail_rec = {
#                     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                     "instance": instance_name,
#                     "beta": float(beta),
#                     "alpha": float(alpha),
#                     "run_id": int(run_id),
#                     "seed": int(seed),
#                     "status": "failed",
#                     "error": err,
#                 }
#                 _append_row_csv(failed_csv, fail_rec)

#                 if not continue_on_error:
#                     raise RuntimeError(err)

#     # -------------------------
#     # 4) 结束后兜底刷新 summary_live
#     # -------------------------
#     if update_summary_each_run and len(progress_records) > 0:
#         df_all = pd.DataFrame(progress_records)
#         df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
#         summary = _build_summary(df_ok)
#         summary.to_csv(summary_live_csv, index=False)

#     # -------------------------
#     # 5) 最终导出 Excel
#     # -------------------------
#     df_all = pd.DataFrame(progress_records)

#     if not df_all.empty:
#         df_ok = df_all[df_all["status"] == "completed"].copy() if "status" in df_all.columns else df_all.copy()
#         summary = _build_summary(df_ok)
#     else:
#         summary = pd.DataFrame()

#     df_pareto = pd.read_csv(pareto_csv) if os.path.exists(pareto_csv) else pd.DataFrame()

#     with pd.ExcelWriter(final_xlsx, engine="openpyxl") as writer:
#         df_all.to_excel(writer, sheet_name="runs", index=False)
#         summary.to_excel(writer, sheet_name="summary", index=False)
#         df_pareto.to_excel(writer, sheet_name="pareto_archive", index=False)

#     print(f"\n[DONE] Progress CSV : {progress_csv}")
#     if update_summary_each_run:
#         print(f"[DONE] Live Summary : {summary_live_csv}")
#     if os.path.exists(failed_csv):
#         print(f"[DONE] Failed Runs  : {failed_csv}")
#     if os.path.exists(pareto_csv):
#         print(f"[DONE] Pareto CSV   : {pareto_csv}")
#     print(f"[DONE] Details Dir   : {details_dir}")
#     print(f"[DONE] Final Excel   : {final_xlsx}")

#     return {
#         "progress_csv": progress_csv,
#         "summary_live_csv": summary_live_csv,
#         "failed_csv": failed_csv,
#         "pareto_csv": pareto_csv,
#         "details_dir": details_dir,
#         "final_xlsx": final_xlsx,
#         "n_total": total_tasks,
#         "n_skipped": skipped,
#         "n_remaining": len(tasks),
#     }






# def _build_summary_spr(df: pd.DataFrame):
#     if df.empty:
#         return pd.DataFrame()

#     agg_dict = {
#         "final_cost": ["count", "mean", "std", "min"],
#         "final_ra": ["mean", "std", "min"],
#         "best_cost_in_archive": ["mean", "std", "min"],
#         "best_ra_in_archive": ["mean", "std", "min"],
#         "archive_size": ["mean", "std", "max"],
#         "runtime_sec": ["mean", "std"],
#         "n_candidates": ["mean", "std"],
#         "n_iterations": ["mean", "std"],
#     }

#     if "t_stage1" in df.columns:
#         agg_dict["t_stage1"] = ["mean", "std"]
#     if "t_stage2" in df.columns:
#         agg_dict["t_stage2"] = ["mean", "std"]

#     summary = (
#         df.groupby(["instance", "beta"])
#           .agg(agg_dict)
#     )

#     summary.columns = ["_".join(col).strip() for col in summary.columns.to_flat_index()]
#     summary = summary.reset_index()

#     if "t_stage2_mean" in summary.columns and "runtime_sec_mean" in summary.columns:
#         summary["stage2_ratio_mean"] = summary["t_stage2_mean"] / summary["runtime_sec_mean"]

#     return summary

# def _run_one_task_spr(args):
#     instance_path, instance_name, beta, run_id, seed = args
#     try:
#         rec = run_once_spr(
#             instance_path=instance_path,
#             beta=beta,
#             run_id=run_id,
#             seed=seed,
#         )
#         return {
#             "ok": True,
#             "instance_name": instance_name,
#             "beta": beta,
#             "run_id": run_id,
#             "seed": seed,
#             "record": rec,
#             "error": None,
#         }
#     except Exception as e:
#         return {
#             "ok": False,
#             "instance_name": instance_name,
#             "beta": beta,
#             "run_id": run_id,
#             "seed": seed,
#             "record": None,
#             "error": repr(e),
#         }
    


    

if __name__ == "__main__":


    # =========================
    # 你自己的全局参数
    # =========================
    BETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ALPHAS = [0.8, 0.85, 0.9, 0.95, 0.99]
    N_RUNS = 10

    # BETAS = [0.1, ]
    # ALPHAS = [0.8, ]
    # N_RUNS = 1

    ## ======== 整个文件夹运行
    input_path = r'D:\02_Research\DataSet\SPR'
    SPR_output_path = r"D:\02_Research\6_experimentResults\SPR_parallel"
    CCP_output_path = r"D:\02_Research\6_experimentResults\CCP_parallel"


    out_spr = run_experiments_parallel_spr(
        BETAS=BETAS,
        N_RUNS=N_RUNS,
        instance_source=input_path,
        results_dir=SPR_output_path,
        base_seed=42,
        max_workers=10,
        update_summary_each_run=True,
        save_every=20,
        continue_on_error=True,
        create_timestamp_subdir=False,
    )

    out_ccp = run_experiments_parallel_ccp(
        BETAS=BETAS,
        ALPHAS=ALPHAS,
        N_RUNS=N_RUNS,
        instance_source=input_path,
        results_dir=CCP_output_path,
        base_seed=42,
        max_workers=10,
        update_summary_each_run=True,
        save_every=20,              # 每完成 10 个成功任务再刷新一次 summary_live
        continue_on_error=True,
        create_timestamp_subdir=False,   # 真正支持续跑时建议 False
    )