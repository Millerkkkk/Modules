import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd


# =========================
# common utils
# =========================
def _list_instances(instance_source: str):
    if os.path.isfile(instance_source):
        return [instance_source]
    return sorted(
        os.path.join(instance_source, f)
        for f in os.listdir(instance_source)
        if f.lower().endswith(".txt")
    )


def _append_row_csv(path: str, record: dict):
    df_row = pd.DataFrame([record])
    write_header = not os.path.exists(path)
    df_row.to_csv(path, mode="a", header=write_header, index=False)


def _append_rows_csv(path: str, rows: list[dict]):
    if not rows:
        return
    df_rows = pd.DataFrame(rows)
    write_header = not os.path.exists(path)
    df_rows.to_csv(path, mode="a", header=write_header, index=False)


def same_point(a, b, eps=1e-9):
    if a is None or b is None:
        return False
    return (
        abs(float(a["cost"]) - float(b["cost"])) <= eps
        and abs(float(a["ra"]) - float(b["ra"])) <= eps
    )


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



def _run_parallel_common(
    *,
    mode_name,
    instance_source,
    results_dir,
    max_workers,
    update_summary_each_run,
    save_every,
    continue_on_error,
    create_timestamp_subdir,
    progress_key_columns,
    make_key,
    make_tasks,
    run_one_task,
    build_summary,
    make_archive_rows,
    save_run_details_pkl,
):
    if not os.path.exists(instance_source):
        raise FileNotFoundError(f"instance_source does not exist: {instance_source}")

    instance_list = _list_instances(instance_source)
    if len(instance_list) == 0:
        raise ValueError(f"No .txt instances found in: {instance_source}")

    if results_dir is None:
        results_dir = os.path.join(os.getcwd(), f"parallel_results_{mode_name}")

    if create_timestamp_subdir:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(results_dir, run_tag)

    os.makedirs(results_dir, exist_ok=True)

    progress_csv = os.path.join(results_dir, "progress_runs.csv")
    summary_live_csv = os.path.join(results_dir, "summary_live.csv")
    failed_csv = os.path.join(results_dir, "failed_runs.csv")
    pareto_csv = os.path.join(results_dir, "pareto_archive.csv")
    final_xlsx = os.path.join(results_dir, "final_results.xlsx")
    details_dir = os.path.join(results_dir, "run_details")

    done = set()
    progress_records = []

    if os.path.exists(progress_csv):
        df_done = pd.read_csv(progress_csv)
        progress_records = df_done.to_dict("records")

        if set(progress_key_columns).issubset(df_done.columns):
            for _, row in df_done.iterrows():
                if "status" not in df_done.columns or row.get("status", "completed") == "completed":
                    done.add(make_key(*[row[c] for c in progress_key_columns]))

        print(f"[RESUME] Loaded progress: {progress_csv} | completed={len(done)}")
    else:
        print(f"[RESUME] No progress file. Will create: {progress_csv}")

    tasks, total_tasks, skipped = make_tasks(instance_list, done)
    print(f"[TASKS] total={total_tasks} | skipped(done)={skipped} | remaining={len(tasks)}")

    if save_every is None or int(save_every) <= 0:
        save_every = 1
    save_every = int(save_every)

    completed_since_last_summary = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(run_one_task, task): task for task in tasks}

        for idx, future in enumerate(as_completed(future_to_task), 1):
            task = future_to_task[future]

            try:
                out = future.result()
            except Exception as e:
                out = {"ok": False, "error": repr(e)}

            if out["ok"]:
                rec = out["record"]
                summary_rec, detail_rec = _split_run_record(rec)

                detail_path = save_run_details_pkl(details_dir, detail_rec=detail_rec, **out)

                summary_rec["timestamp"] = summary_rec.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                summary_rec["status"] = "completed"
                summary_rec["details_path"] = detail_path

                _append_row_csv(progress_csv, summary_rec)
                progress_records.append(summary_rec)

                archive_rows = make_archive_rows(rec)
                _append_rows_csv(pareto_csv, archive_rows)

                completed_since_last_summary += 1
                if update_summary_each_run and completed_since_last_summary >= save_every:
                    df_all = pd.DataFrame(progress_records)
                    df_ok = df_all[df_all["status"] == "completed"].copy()
                    summary = build_summary(df_ok)
                    summary.to_csv(summary_live_csv, index=False)
                    completed_since_last_summary = 0
            else:
                fail_rec = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "status": "failed", "error": out["error"]}
                fail_rec.update({k: v for k, v in out.items() if k not in {"ok", "record", "error"}})
                _append_row_csv(failed_csv, fail_rec)

                if not continue_on_error:
                    raise RuntimeError(out["error"])

    if update_summary_each_run and len(progress_records) > 0:
        df_all = pd.DataFrame(progress_records)
        df_ok = df_all[df_all["status"] == "completed"].copy()
        summary = build_summary(df_ok)
        summary.to_csv(summary_live_csv, index=False)

    df_all = pd.DataFrame(progress_records)
    summary = build_summary(df_all[df_all["status"] == "completed"].copy()) if not df_all.empty else pd.DataFrame()
    df_pareto = pd.read_csv(pareto_csv) if os.path.exists(pareto_csv) else pd.DataFrame()

    with pd.ExcelWriter(final_xlsx, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="runs", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        df_pareto.to_excel(writer, sheet_name="pareto_archive", index=False)

    return {
        "progress_csv": progress_csv,
        "summary_live_csv": summary_live_csv,
        "failed_csv": failed_csv,
        "pareto_csv": pareto_csv,
        "details_dir": details_dir,
        "final_xlsx": final_xlsx,
        "n_total": total_tasks,
        "n_skipped": skipped,
        "n_remaining": len(tasks),
    }