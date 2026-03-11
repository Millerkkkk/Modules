# =========================
# SPR adapter
# =========================
import os
import pickle
import pandas as pd

from Experiment.common import same_point
from main_SPR_SD_MOP import run_once_spr


def _make_key_spr(instance: str, beta: float, run_id: int):
    return (str(instance), round(float(beta), 10), int(run_id))


def _run_one_task_spr(args):
    instance_path, instance_name, beta, run_id, seed = args
    try:
        rec = run_once_spr(
            instance_path=instance_path,
            beta=beta,
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


def _build_summary_spr(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame()

    agg_dict = {
        "final_cost": ["count", "mean", "std", "min"],
        "final_ra": ["mean", "std", "min"],
        "best_cost_in_archive": ["mean", "std", "min"],
        "best_ra_in_archive": ["mean", "std", "min"],
        "archive_size": ["mean", "std", "max"],
        "runtime_sec": ["mean", "std"],
        "n_candidates": ["mean", "std"],
        "n_iterations": ["mean", "std"],
    }
    if "t_stage1" in df.columns:
        agg_dict["t_stage1"] = ["mean", "std"]
    if "t_stage2" in df.columns:
        agg_dict["t_stage2"] = ["mean", "std"]
    if "saa_S" in df.columns:
        agg_dict["saa_S"] = ["mean"]
    if "proxy_S" in df.columns:
        agg_dict["proxy_S"] = ["mean"]

    summary = df.groupby(["instance", "beta"]).agg(agg_dict)
    summary.columns = ["_".join(col).strip() for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    return summary


def _make_archive_rows_spr(rec: dict):
    archive = rec.get("archive", [])
    final_best = rec.get("final_best", None)
    min_cost_sol = rec.get("min_cost_sol", None)
    min_ra_sol = rec.get("min_ra_sol", None)

    rows = []
    for i, s in enumerate(archive):
        total_ra = float(s.get("ra", 0.0))
        stage1_ra = float(s.get("stage1_ra", 0.0))
        stage2_ra = float(s.get("stage2_ra", 0.0))

        rows.append({
            "timestamp": rec.get("timestamp"),
            "instance": rec.get("instance"),
            "beta": float(rec.get("beta")),
            "run_id": int(rec.get("run_id")),
            "seed": int(rec.get("seed")),
            "sol_id": int(i),

            # 总目标
            "cost": float(s["cost"]),
            "ra": total_ra,

            # 两阶段 cost
            "stage1_cost": float(s.get("stage1_cost", 0.0)),
            "stage2_cost": float(s.get("stage2_cost", 0.0)),

            # 两阶段 RA
            "stage1_ra": stage1_ra,
            "stage2_ra": stage2_ra,

            # 方便分析的占比
            "stage2_cost_ratio": float(s.get("stage2_cost", 0.0)) / (float(s["cost"]) + 1e-9),
            "stage2_ra_ratio": stage2_ra / (total_ra + 1e-9),

            # 标记
            "is_final_best": bool(same_point(s, final_best)) if final_best is not None else False,
            "is_min_cost": bool(same_point(s, min_cost_sol)) if min_cost_sol is not None else False,
            "is_min_ra": bool(same_point(s, min_ra_sol)) if min_ra_sol is not None else False,
        })
    return rows


def _save_run_details_pkl_spr(details_dir, instance_name, beta, run_id, detail_rec):
    os.makedirs(details_dir, exist_ok=True)
    fname = f"{instance_name}_beta{beta:.6f}_run{run_id}.pkl"
    path = os.path.join(details_dir, fname)
    with open(path, "wb") as f:
        pickle.dump(detail_rec, f)
    return path


def _make_tasks_spr(instance_list, BETAS, N_RUNS, base_seed, done):
    tasks = []
    total_tasks = 0
    skipped = 0

    for instance_path in instance_list:
        instance_name = os.path.splitext(os.path.basename(instance_path))[0]
        name_code = sum(ord(c) for c in instance_name)

        for beta in BETAS:
            for run_id in range(1, N_RUNS + 1):
                total_tasks += 1
                key = _make_key_spr(instance_name, beta, run_id)
                if key in done:
                    skipped += 1
                    continue

                seed = int(base_seed) + 100000 * name_code + 1000 * int(round(float(beta) * 100)) + (run_id - 1)
                tasks.append((instance_path, instance_name, float(beta), int(run_id), int(seed)))

    return tasks, total_tasks, skipped