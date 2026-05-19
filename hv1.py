import os
import re
import json
import copy
from collections import defaultdict

import numpy as np
import pandas as pd


def dominates_min(a, b, eps=1e-12):
    return (
        a[0] <= b[0] + eps
        and a[1] <= b[1] + eps
        and (a[0] < b[0] - eps or a[1] < b[1] - eps)
    )


def nondominated_filter(points, eps=1e-12):
    unique_points = []
    seen = set()

    for p in points:
        c = float(p[0])
        r = float(p[1])
        key = (round(c, 12), round(r, 12))
        if key not in seen:
            seen.add(key)
            unique_points.append((c, r))

    nd = []
    for i, p in enumerate(unique_points):
        dominated = False
        for j, q in enumerate(unique_points):
            if i == j:
                continue
            if dominates_min(q, p, eps=eps):
                dominated = True
                break
        if not dominated:
            nd.append(p)

    return nd


def hypervolume_2d_min(points, ref_point=(1.1, 1.1), eps=1e-12):
    rx, ry = float(ref_point[0]), float(ref_point[1])

    filtered = []
    for c, r in points:
        c = float(c)
        r = float(r)
        if c <= rx + eps and r <= ry + eps:
            filtered.append((c, r))

    if not filtered:
        return 0.0

    nd = nondominated_filter(filtered, eps=eps)
    nd = sorted(nd, key=lambda x: (x[0], x[1]))

    hv = 0.0
    prev_r = ry

    for c, r in nd:
        width = max(rx - c, 0.0)
        height = max(prev_r - r, 0.0)
        hv += width * height
        prev_r = min(prev_r, r)

    return float(hv)


def normalize_points(points, cmin, cmax, rmin, rmax, eps=1e-12):
    dc = max(cmax - cmin, eps)
    dr = max(rmax - rmin, eps)

    normed = []
    for c, r in points:
        cn = (float(c) - cmin) / dc
        rn = (float(r) - rmin) / dr
        normed.append((cn, rn))
    return normed


def igd_plus(reference_front, approx_front):
    if not reference_front:
        return np.nan
    if not approx_front:
        return np.inf

    ref_arr = np.asarray(reference_front, dtype=float)
    app_arr = np.asarray(approx_front, dtype=float)

    distances = []
    for z in ref_arr:
        diff = np.maximum(app_arr - z, 0.0)
        d = np.sqrt(np.sum(diff * diff, axis=1))
        distances.append(np.min(d))

    return float(np.mean(distances))


def load_points_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pts = data.get("pareto_points", None)
    if pts is not None:
        clean_pts = []
        for item in pts:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                clean_pts.append((float(item[0]), float(item[1])))
        if clean_pts:
            return data, clean_pts

    archive = data.get("archive", None)
    if archive is not None:
        clean_pts = []
        for sol in archive:
            if isinstance(sol, dict) and ("cost" in sol) and ("ra" in sol):
                clean_pts.append((float(sol["cost"]), float(sol["ra"])))
        if clean_pts:
            return data, clean_pts

    raise ValueError(f"No valid pareto_points/archive found in: {json_path}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sanitize_label(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def extract_group_label(root, base_path, folder_level=1):
    """
    从 base_path 下的相对路径中提取第 folder_level 层文件夹名作为 group_label。

    规则：
    1. 如果 root == base_path，说明文件就在输入根目录下，返回 base_path 文件夹名；
    2. folder_level 从 1 开始计数；
    3. 如果相对路径层数不足，返回最后一层目录名。
    """
    rel_path = os.path.relpath(root, base_path)

    # 文件直接位于 base_path 下
    if rel_path == ".":
        return os.path.basename(os.path.normpath(base_path))

    parts = rel_path.split(os.sep)

    # folder_level 非法时，默认取第一层
    if folder_level < 1:
        folder_level = 1

    # 层数不足时，取最后一层
    if len(parts) < folder_level:
        return parts[-1]

    return parts[folder_level - 1]


def main(path, out_path, folder_level=1):
    ensure_dir(out_path)
    out_json_dir = os.path.join(out_path, "json_with_unified_metrics")
    ensure_dir(out_json_dir)

    file_pattern = re.compile(
        r"^(?P<instance>.+?)_beta(?P<beta>\d+(?:\.\d+)?)_run(?P<run>\d+)\.json$",
        re.IGNORECASE,
    )

    files_by_instance = defaultdict(list)
    skipped_files = []

    # 1) 扫描所有 json，并从文件夹名提取 group_label
    for root, dirs, files in os.walk(path):
        group_label = extract_group_label(root, path, folder_level=folder_level)

        for fn in files:
            if not fn.lower().endswith(".json"):
                continue

            full_path = os.path.join(root, fn)

            if group_label is None:
                skipped_files.append((full_path, "cannot_extract_group_label"))
                continue

            m = file_pattern.match(fn)
            if not m:
                skipped_files.append((full_path, "filename_not_matched"))
                continue

            instance = m.group("instance")
            beta = float(m.group("beta"))
            run = int(m.group("run"))

            files_by_instance[instance].append(
                {
                    "filename": fn,
                    "path": full_path,
                    "instance": instance,
                    "beta": beta,
                    "run": run,
                    "group_label": group_label,
                    "root_folder": root,
                }
            )

    if not files_by_instance:
        raise RuntimeError("No valid json files found.")

    per_run_rows = []
    norm_info_rows = []
    ref_front_rows = []

    # 2) 按 instance 统一计算 normalization + reference front
    for instance, items in sorted(files_by_instance.items()):
        union_points = []

        for item in items:
            try:
                data, pts = load_points_from_json(item["path"])
                item["data"] = data
                item["points_raw"] = pts
                union_points.extend(pts)
            except Exception as e:
                item["load_error"] = str(e)

        valid_items = [x for x in items if "points_raw" in x]
        invalid_items = [x for x in items if "points_raw" not in x]

        for bad in invalid_items:
            skipped_files.append((bad["path"], bad.get("load_error", "unknown_error")))

        if not valid_items:
            continue

        all_costs = [float(c) for c, _ in union_points]
        all_ras = [float(r) for _, r in union_points]

        cmin = min(all_costs)
        cmax = max(all_costs)
        rmin = min(all_ras)
        rmax = max(all_ras)

        union_points_norm = normalize_points(union_points, cmin, cmax, rmin, rmax)
        reference_front = nondominated_filter(union_points_norm)
        reference_front = sorted(reference_front, key=lambda x: (x[0], x[1]))

        norm_info_rows.append(
            {
                "instance": instance,
                "cost_min_instance": cmin,
                "cost_max_instance": cmax,
                "ra_min_instance": rmin,
                "ra_max_instance": rmax,
                "n_files": len(valid_items),
                "n_union_points_raw": len(union_points),
                "n_reference_front_points": len(reference_front),
            }
        )

        for idx, (c, r) in enumerate(reference_front, start=1):
            ref_front_rows.append(
                {
                    "instance": instance,
                    "ref_point_id": idx,
                    "cost_norm": float(c),
                    "ra_norm": float(r),
                }
            )

        # 3) 每个 run 重算 HV / IGD+，并新增 best cost / best ra / pareto size
        for item in valid_items:
            raw_points = item["points_raw"]
            nd_raw = nondominated_filter(raw_points)

            # 原始点中的最小值
            min_cost_raw = min((c for c, r in raw_points), default=np.nan)
            min_ra_raw = min((r for c, r in raw_points), default=np.nan)

            # Pareto 非支配集中的最小值
            min_cost_pareto = min((c for c, r in nd_raw), default=np.nan)
            min_ra_pareto = min((r for c, r in nd_raw), default=np.nan)

            # Pareto 个数
            pareto_size = len(nd_raw)

            norm_points = normalize_points(nd_raw, cmin, cmax, rmin, rmax)
            nd_norm = nondominated_filter(norm_points)
            nd_norm = sorted(nd_norm, key=lambda x: (x[0], x[1]))

            hv_unified = hypervolume_2d_min(nd_norm, ref_point=(1, 1))
            igd_plus_unified = igd_plus(reference_front, nd_norm)

            row = {
                "instance": item["instance"],
                "beta": item["beta"],
                "run": item["run"],
                "group_label": item["group_label"],
                "filename": item["filename"],
                "filepath": item["path"],

                "n_points_raw": len(raw_points),
                "n_points_nd_raw": len(nd_raw),   # 保留原字段
                "n_points_nd_norm": len(nd_norm),

                "pareto_size": int(pareto_size),

                "min_cost_raw": float(min_cost_raw),
                "min_ra_raw": float(min_ra_raw),

                "min_cost_pareto": float(min_cost_pareto),
                "min_ra_pareto": float(min_ra_pareto),

                "n_reference_front_points": len(reference_front),
                "hv_unified": float(hv_unified),
                "igd_plus_unified": float(igd_plus_unified),
                "cost_min_instance": float(cmin),
                "cost_max_instance": float(cmax),
                "ra_min_instance": float(rmin),
                "ra_max_instance": float(rmax),
            }
            per_run_rows.append(row)

            new_data = copy.deepcopy(item["data"])
            new_data["hv_unified"] = float(hv_unified)
            new_data["igd_plus_unified"] = float(igd_plus_unified)
            new_data["unified_metrics_ref_point_hv"] = [1.1, 1.1]
            new_data["unified_metrics_normalization"] = {
                "instance": instance,
                "cost_min_instance": float(cmin),
                "cost_max_instance": float(cmax),
                "ra_min_instance": float(rmin),
                "ra_max_instance": float(rmax),
            }
            new_data["reference_front_size"] = int(len(reference_front))
            new_data["group_label"] = item["group_label"]

            # 新增写回 json 的统计值
            new_data["pareto_size"] = int(pareto_size)
            new_data["min_cost_raw"] = float(min_cost_raw)
            new_data["min_ra_raw"] = float(min_ra_raw)
            new_data["min_cost_pareto"] = float(min_cost_pareto)
            new_data["min_ra_pareto"] = float(min_ra_pareto)

            out_json_name = (
                f"{item['instance']}_beta{item['beta']:.3f}_{sanitize_label(item['group_label'])}_run{item['run']}.json"
            )
            out_json_path = os.path.join(out_json_dir, out_json_name)

            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)

    df_run = pd.DataFrame(per_run_rows)
    if df_run.empty:
        raise RuntimeError("No valid results were processed.")

    df_run = df_run.sort_values(by=["instance", "group_label", "beta", "run"]).reset_index(drop=True)
    df_run.to_csv(
        os.path.join(out_path, "hv_igd_per_run.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 4) summary: instance + group + beta
    grouped_instance_group_beta = (
        df_run.groupby(["instance", "group_label", "beta"], as_index=False)
        .agg(
            hv_unified_count=("hv_unified", "count"),
            hv_unified_mean=("hv_unified", "mean"),
            hv_unified_std=("hv_unified", "std"),
            hv_unified_min=("hv_unified", "min"),
            hv_unified_median=("hv_unified", "median"),
            hv_unified_max=("hv_unified", "max"),

            igd_plus_unified_count=("igd_plus_unified", "count"),
            igd_plus_unified_mean=("igd_plus_unified", "mean"),
            igd_plus_unified_std=("igd_plus_unified", "std"),
            igd_plus_unified_min=("igd_plus_unified", "min"),
            igd_plus_unified_median=("igd_plus_unified", "median"),
            igd_plus_unified_max=("igd_plus_unified", "max"),

            pareto_size_count=("pareto_size", "count"),
            pareto_size_mean=("pareto_size", "mean"),
            pareto_size_std=("pareto_size", "std"),
            pareto_size_min=("pareto_size", "min"),
            pareto_size_median=("pareto_size", "median"),
            pareto_size_max=("pareto_size", "max"),

            min_cost_pareto_count=("min_cost_pareto", "count"),
            min_cost_pareto_mean=("min_cost_pareto", "mean"),
            min_cost_pareto_std=("min_cost_pareto", "std"),
            min_cost_pareto_min=("min_cost_pareto", "min"),
            min_cost_pareto_median=("min_cost_pareto", "median"),
            min_cost_pareto_max=("min_cost_pareto", "max"),

            min_ra_pareto_count=("min_ra_pareto", "count"),
            min_ra_pareto_mean=("min_ra_pareto", "mean"),
            min_ra_pareto_std=("min_ra_pareto", "std"),
            min_ra_pareto_min=("min_ra_pareto", "min"),
            min_ra_pareto_median=("min_ra_pareto", "median"),
            min_ra_pareto_max=("min_ra_pareto", "max"),

            min_cost_raw_count=("min_cost_raw", "count"),
            min_cost_raw_mean=("min_cost_raw", "mean"),
            min_cost_raw_std=("min_cost_raw", "std"),
            min_cost_raw_min=("min_cost_raw", "min"),
            min_cost_raw_median=("min_cost_raw", "median"),
            min_cost_raw_max=("min_cost_raw", "max"),

            min_ra_raw_count=("min_ra_raw", "count"),
            min_ra_raw_mean=("min_ra_raw", "mean"),
            min_ra_raw_std=("min_ra_raw", "std"),
            min_ra_raw_min=("min_ra_raw", "min"),
            min_ra_raw_median=("min_ra_raw", "median"),
            min_ra_raw_max=("min_ra_raw", "max"),

            n_points_nd_norm_mean=("n_points_nd_norm", "mean"),
            n_reference_front_points=("n_reference_front_points", "mean"),
        )
    )
    grouped_instance_group_beta.to_csv(
        os.path.join(out_path, "hv_igd_summary_by_instance_group_beta.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 5) summary: instance + group
    grouped_instance_group = (
        df_run.groupby(["instance", "group_label"], as_index=False)
        .agg(
            hv_unified_count=("hv_unified", "count"),
            hv_unified_mean=("hv_unified", "mean"),
            hv_unified_std=("hv_unified", "std"),
            hv_unified_min=("hv_unified", "min"),
            hv_unified_median=("hv_unified", "median"),
            hv_unified_max=("hv_unified", "max"),

            igd_plus_unified_count=("igd_plus_unified", "count"),
            igd_plus_unified_mean=("igd_plus_unified", "mean"),
            igd_plus_unified_std=("igd_plus_unified", "std"),
            igd_plus_unified_min=("igd_plus_unified", "min"),
            igd_plus_unified_median=("igd_plus_unified", "median"),
            igd_plus_unified_max=("igd_plus_unified", "max"),

            pareto_size_count=("pareto_size", "count"),
            pareto_size_mean=("pareto_size", "mean"),
            pareto_size_std=("pareto_size", "std"),
            pareto_size_min=("pareto_size", "min"),
            pareto_size_median=("pareto_size", "median"),
            pareto_size_max=("pareto_size", "max"),

            min_cost_pareto_count=("min_cost_pareto", "count"),
            min_cost_pareto_mean=("min_cost_pareto", "mean"),
            min_cost_pareto_std=("min_cost_pareto", "std"),
            min_cost_pareto_min=("min_cost_pareto", "min"),
            min_cost_pareto_median=("min_cost_pareto", "median"),
            min_cost_pareto_max=("min_cost_pareto", "max"),

            min_ra_pareto_count=("min_ra_pareto", "count"),
            min_ra_pareto_mean=("min_ra_pareto", "mean"),
            min_ra_pareto_std=("min_ra_pareto", "std"),
            min_ra_pareto_min=("min_ra_pareto", "min"),
            min_ra_pareto_median=("min_ra_pareto", "median"),
            min_ra_pareto_max=("min_ra_pareto", "max"),

            min_cost_raw_count=("min_cost_raw", "count"),
            min_cost_raw_mean=("min_cost_raw", "mean"),
            min_cost_raw_std=("min_cost_raw", "std"),
            min_cost_raw_min=("min_cost_raw", "min"),
            min_cost_raw_median=("min_cost_raw", "median"),
            min_cost_raw_max=("min_cost_raw", "max"),

            min_ra_raw_count=("min_ra_raw", "count"),
            min_ra_raw_mean=("min_ra_raw", "mean"),
            min_ra_raw_std=("min_ra_raw", "std"),
            min_ra_raw_min=("min_ra_raw", "min"),
            min_ra_raw_median=("min_ra_raw", "median"),
            min_ra_raw_max=("min_ra_raw", "max"),
        )
    )
    grouped_instance_group.to_csv(
        os.path.join(out_path, "hv_igd_summary_by_instance_group.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 6) summary: group overall
    grouped_group = (
        df_run.groupby(["group_label"], as_index=False)
        .agg(
            hv_unified_count=("hv_unified", "count"),
            hv_unified_mean=("hv_unified", "mean"),
            hv_unified_std=("hv_unified", "std"),
            hv_unified_min=("hv_unified", "min"),
            hv_unified_median=("hv_unified", "median"),
            hv_unified_max=("hv_unified", "max"),

            igd_plus_unified_count=("igd_plus_unified", "count"),
            igd_plus_unified_mean=("igd_plus_unified", "mean"),
            igd_plus_unified_std=("igd_plus_unified", "std"),
            igd_plus_unified_min=("igd_plus_unified", "min"),
            igd_plus_unified_median=("igd_plus_unified", "median"),
            igd_plus_unified_max=("igd_plus_unified", "max"),

            pareto_size_count=("pareto_size", "count"),
            pareto_size_mean=("pareto_size", "mean"),
            pareto_size_std=("pareto_size", "std"),
            pareto_size_min=("pareto_size", "min"),
            pareto_size_median=("pareto_size", "median"),
            pareto_size_max=("pareto_size", "max"),

            min_cost_pareto_count=("min_cost_pareto", "count"),
            min_cost_pareto_mean=("min_cost_pareto", "mean"),
            min_cost_pareto_std=("min_cost_pareto", "std"),
            min_cost_pareto_min=("min_cost_pareto", "min"),
            min_cost_pareto_median=("min_cost_pareto", "median"),
            min_cost_pareto_max=("min_cost_pareto", "max"),

            min_ra_pareto_count=("min_ra_pareto", "count"),
            min_ra_pareto_mean=("min_ra_pareto", "mean"),
            min_ra_pareto_std=("min_ra_pareto", "std"),
            min_ra_pareto_min=("min_ra_pareto", "min"),
            min_ra_pareto_median=("min_ra_pareto", "median"),
            min_ra_pareto_max=("min_ra_pareto", "max"),

            min_cost_raw_count=("min_cost_raw", "count"),
            min_cost_raw_mean=("min_cost_raw", "mean"),
            min_cost_raw_std=("min_cost_raw", "std"),
            min_cost_raw_min=("min_cost_raw", "min"),
            min_cost_raw_median=("min_cost_raw", "median"),
            min_cost_raw_max=("min_cost_raw", "max"),

            min_ra_raw_count=("min_ra_raw", "count"),
            min_ra_raw_mean=("min_ra_raw", "mean"),
            min_ra_raw_std=("min_ra_raw", "std"),
            min_ra_raw_min=("min_ra_raw", "min"),
            min_ra_raw_median=("min_ra_raw", "median"),
            min_ra_raw_max=("min_ra_raw", "max"),
        )
    )
    grouped_group.to_csv(
        os.path.join(out_path, "hv_igd_summary_by_group_overall.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(norm_info_rows).to_csv(
        os.path.join(out_path, "normalization_info.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(ref_front_rows).to_csv(
        os.path.join(out_path, "reference_front_by_instance.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    if skipped_files:
        pd.DataFrame(skipped_files, columns=["path_or_file", "reason"]).to_csv(
            os.path.join(out_path, "skipped_files.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    print("=" * 90)
    print("Done.")
    print(f"Input path:  {path}")
    print(f"Output path: {out_path}")
    print(f"Processed files: {len(df_run)}")
    print(f"Instances: {df_run['instance'].nunique()}")
    print(f"Groups: {sorted(df_run['group_label'].unique().tolist())}")
    print(f"Betas: {sorted(df_run['beta'].unique().tolist())}")
    print("=" * 90)


def main1(path, out_path, folder_level=1):
    ensure_dir(out_path)
    out_json_dir = os.path.join(out_path, "json_with_unified_metrics")
    ensure_dir(out_json_dir)

    file_pattern = re.compile(
        r"^(?P<stem>.+?)_beta(?P<beta>\d+(?:\.\d+)?)_run(?P<run>\d+)\.json$",
        re.IGNORECASE,
    )

    algo_suffixes = {"cmopso", "moead", "nsga2", "nsga3"}

    files_by_instance = defaultdict(list)
    skipped_files = []

    # 1) 扫描所有 json，并从文件夹名提取 group_label
    for root, dirs, files in os.walk(path):
        group_label = extract_group_label(root, path, folder_level=folder_level)

        for fn in files:
            if not fn.lower().endswith(".json"):
                continue

            full_path = os.path.join(root, fn)

            if group_label is None:
                skipped_files.append((full_path, "cannot_extract_group_label"))
                continue

            m = file_pattern.match(fn)
            if not m:
                skipped_files.append((full_path, "filename_not_matched"))
                continue

            stem = m.group("stem")
            beta = float(m.group("beta"))
            run = int(m.group("run"))

            parts = stem.split("_")
            if parts and parts[-1].lower() in algo_suffixes:
                instance = "_".join(parts[:-1])
                algo_from_filename = parts[-1].lower()
            else:
                instance = stem
                algo_from_filename = None

            files_by_instance[instance].append(
                {
                    "filename": fn,
                    "path": full_path,
                    "instance": instance,
                    "beta": beta,
                    "run": run,
                    "group_label": group_label,
                    "algo_from_filename": algo_from_filename,
                    "root_folder": root,
                }
            )

    if not files_by_instance:
        raise RuntimeError("No valid json files found.")

    per_run_rows = []
    norm_info_rows = []
    ref_front_rows = []

    # 2) 按 instance 统一计算 normalization + reference front
    for instance, items in sorted(files_by_instance.items()):
        union_points = []

        for item in items:
            try:
                data, pts = load_points_from_json(item["path"])
                item["data"] = data
                item["points_raw"] = pts
                union_points.extend(pts)
            except Exception as e:
                item["load_error"] = str(e)

        valid_items = [x for x in items if "points_raw" in x]
        invalid_items = [x for x in items if "points_raw" not in x]

        for bad in invalid_items:
            skipped_files.append((bad["path"], bad.get("load_error", "unknown_error")))

        if not valid_items:
            continue

        all_costs = [float(c) for c, _ in union_points]
        all_ras = [float(r) for _, r in union_points]

        cmin = min(all_costs)
        cmax = max(all_costs)
        rmin = min(all_ras)
        rmax = max(all_ras)

        union_points_norm = normalize_points(union_points, cmin, cmax, rmin, rmax)
        reference_front = nondominated_filter(union_points_norm)
        reference_front = sorted(reference_front, key=lambda x: (x[0], x[1]))

        norm_info_rows.append(
            {
                "instance": instance,
                "cost_min_instance": cmin,
                "cost_max_instance": cmax,
                "ra_min_instance": rmin,
                "ra_max_instance": rmax,
                "n_files": len(valid_items),
                "n_union_points_raw": len(union_points),
                "n_reference_front_points": len(reference_front),
            }
        )

        for idx, (c, r) in enumerate(reference_front, start=1):
            ref_front_rows.append(
                {
                    "instance": instance,
                    "ref_point_id": idx,
                    "cost_norm": float(c),
                    "ra_norm": float(r),
                }
            )

        # 3) 每个 run 重算 HV / IGD+，并新增 best cost / best ra / pareto size
        for item in valid_items:
            raw_points = item["points_raw"]
            nd_raw = nondominated_filter(raw_points)

            # 原始点中的最小值
            min_cost_raw = min((c for c, r in raw_points), default=np.nan)
            min_ra_raw = min((r for c, r in raw_points), default=np.nan)

            # Pareto 非支配集中的最小值
            min_cost_pareto = min((c for c, r in nd_raw), default=np.nan)
            min_ra_pareto = min((r for c, r in nd_raw), default=np.nan)

            # Pareto 个数
            pareto_size = len(nd_raw)

            norm_points = normalize_points(nd_raw, cmin, cmax, rmin, rmax)
            nd_norm = nondominated_filter(norm_points)
            nd_norm = sorted(nd_norm, key=lambda x: (x[0], x[1]))

            hv_unified = hypervolume_2d_min(nd_norm, ref_point=(1, 1))
            igd_plus_unified = igd_plus(reference_front, nd_norm)

            row = {
                "instance": item["instance"],
                "beta": item["beta"],
                "run": item["run"],
                "group_label": item["group_label"],
                "filename": item["filename"],
                "filepath": item["path"],

                "n_points_raw": len(raw_points),
                "n_points_nd_raw": len(nd_raw),   # 保留原字段
                "n_points_nd_norm": len(nd_norm),

                "pareto_size": int(pareto_size),

                "min_cost_raw": float(min_cost_raw),
                "min_ra_raw": float(min_ra_raw),

                "min_cost_pareto": float(min_cost_pareto),
                "min_ra_pareto": float(min_ra_pareto),

                "n_reference_front_points": len(reference_front),
                "hv_unified": float(hv_unified),
                "igd_plus_unified": float(igd_plus_unified),
                "cost_min_instance": float(cmin),
                "cost_max_instance": float(cmax),
                "ra_min_instance": float(rmin),
                "ra_max_instance": float(rmax),
            }
            per_run_rows.append(row)

            new_data = copy.deepcopy(item["data"])
            new_data["hv_unified"] = float(hv_unified)
            new_data["igd_plus_unified"] = float(igd_plus_unified)
            new_data["unified_metrics_ref_point_hv"] = [1.1, 1.1]
            new_data["unified_metrics_normalization"] = {
                "instance": instance,
                "cost_min_instance": float(cmin),
                "cost_max_instance": float(cmax),
                "ra_min_instance": float(rmin),
                "ra_max_instance": float(rmax),
            }
            new_data["reference_front_size"] = int(len(reference_front))
            new_data["group_label"] = item["group_label"]

            # 新增写回 json 的统计值
            new_data["pareto_size"] = int(pareto_size)
            new_data["min_cost_raw"] = float(min_cost_raw)
            new_data["min_ra_raw"] = float(min_ra_raw)
            new_data["min_cost_pareto"] = float(min_cost_pareto)
            new_data["min_ra_pareto"] = float(min_ra_pareto)

            out_json_name = (
                f"{item['instance']}_beta{item['beta']:.3f}_{sanitize_label(item['group_label'])}_run{item['run']}.json"
            )
            out_json_path = os.path.join(out_json_dir, out_json_name)

            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)

    df_run = pd.DataFrame(per_run_rows)
    if df_run.empty:
        raise RuntimeError("No valid results were processed.")

    df_run = df_run.sort_values(by=["instance", "group_label", "beta", "run"]).reset_index(drop=True)
    df_run.to_csv(
        os.path.join(out_path, "hv_igd_per_run.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 4) summary: instance + group + beta
    grouped_instance_group_beta = (
        df_run.groupby(["instance", "group_label", "beta"], as_index=False)
        .agg(
            hv_unified_count=("hv_unified", "count"),
            hv_unified_mean=("hv_unified", "mean"),
            hv_unified_std=("hv_unified", "std"),
            hv_unified_min=("hv_unified", "min"),
            hv_unified_median=("hv_unified", "median"),
            hv_unified_max=("hv_unified", "max"),

            igd_plus_unified_count=("igd_plus_unified", "count"),
            igd_plus_unified_mean=("igd_plus_unified", "mean"),
            igd_plus_unified_std=("igd_plus_unified", "std"),
            igd_plus_unified_min=("igd_plus_unified", "min"),
            igd_plus_unified_median=("igd_plus_unified", "median"),
            igd_plus_unified_max=("igd_plus_unified", "max"),

            pareto_size_count=("pareto_size", "count"),
            pareto_size_mean=("pareto_size", "mean"),
            pareto_size_std=("pareto_size", "std"),
            pareto_size_min=("pareto_size", "min"),
            pareto_size_median=("pareto_size", "median"),
            pareto_size_max=("pareto_size", "max"),

            min_cost_pareto_count=("min_cost_pareto", "count"),
            min_cost_pareto_mean=("min_cost_pareto", "mean"),
            min_cost_pareto_std=("min_cost_pareto", "std"),
            min_cost_pareto_min=("min_cost_pareto", "min"),
            min_cost_pareto_median=("min_cost_pareto", "median"),
            min_cost_pareto_max=("min_cost_pareto", "max"),

            min_ra_pareto_count=("min_ra_pareto", "count"),
            min_ra_pareto_mean=("min_ra_pareto", "mean"),
            min_ra_pareto_std=("min_ra_pareto", "std"),
            min_ra_pareto_min=("min_ra_pareto", "min"),
            min_ra_pareto_median=("min_ra_pareto", "median"),
            min_ra_pareto_max=("min_ra_pareto", "max"),

            min_cost_raw_count=("min_cost_raw", "count"),
            min_cost_raw_mean=("min_cost_raw", "mean"),
            min_cost_raw_std=("min_cost_raw", "std"),
            min_cost_raw_min=("min_cost_raw", "min"),
            min_cost_raw_median=("min_cost_raw", "median"),
            min_cost_raw_max=("min_cost_raw", "max"),

            min_ra_raw_count=("min_ra_raw", "count"),
            min_ra_raw_mean=("min_ra_raw", "mean"),
            min_ra_raw_std=("min_ra_raw", "std"),
            min_ra_raw_min=("min_ra_raw", "min"),
            min_ra_raw_median=("min_ra_raw", "median"),
            min_ra_raw_max=("min_ra_raw", "max"),

            n_points_nd_norm_mean=("n_points_nd_norm", "mean"),
            n_reference_front_points=("n_reference_front_points", "mean"),
        )
    )
    grouped_instance_group_beta.to_csv(
        os.path.join(out_path, "hv_igd_summary_by_instance_group_beta.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 5) summary: instance + group
    grouped_instance_group = (
        df_run.groupby(["instance", "group_label"], as_index=False)
        .agg(
            hv_unified_count=("hv_unified", "count"),
            hv_unified_mean=("hv_unified", "mean"),
            hv_unified_std=("hv_unified", "std"),
            hv_unified_min=("hv_unified", "min"),
            hv_unified_median=("hv_unified", "median"),
            hv_unified_max=("hv_unified", "max"),

            igd_plus_unified_count=("igd_plus_unified", "count"),
            igd_plus_unified_mean=("igd_plus_unified", "mean"),
            igd_plus_unified_std=("igd_plus_unified", "std"),
            igd_plus_unified_min=("igd_plus_unified", "min"),
            igd_plus_unified_median=("igd_plus_unified", "median"),
            igd_plus_unified_max=("igd_plus_unified", "max"),

            pareto_size_count=("pareto_size", "count"),
            pareto_size_mean=("pareto_size", "mean"),
            pareto_size_std=("pareto_size", "std"),
            pareto_size_min=("pareto_size", "min"),
            pareto_size_median=("pareto_size", "median"),
            pareto_size_max=("pareto_size", "max"),

            min_cost_pareto_count=("min_cost_pareto", "count"),
            min_cost_pareto_mean=("min_cost_pareto", "mean"),
            min_cost_pareto_std=("min_cost_pareto", "std"),
            min_cost_pareto_min=("min_cost_pareto", "min"),
            min_cost_pareto_median=("min_cost_pareto", "median"),
            min_cost_pareto_max=("min_cost_pareto", "max"),

            min_ra_pareto_count=("min_ra_pareto", "count"),
            min_ra_pareto_mean=("min_ra_pareto", "mean"),
            min_ra_pareto_std=("min_ra_pareto", "std"),
            min_ra_pareto_min=("min_ra_pareto", "min"),
            min_ra_pareto_median=("min_ra_pareto", "median"),
            min_ra_pareto_max=("min_ra_pareto", "max"),

            min_cost_raw_count=("min_cost_raw", "count"),
            min_cost_raw_mean=("min_cost_raw", "mean"),
            min_cost_raw_std=("min_cost_raw", "std"),
            min_cost_raw_min=("min_cost_raw", "min"),
            min_cost_raw_median=("min_cost_raw", "median"),
            min_cost_raw_max=("min_cost_raw", "max"),

            min_ra_raw_count=("min_ra_raw", "count"),
            min_ra_raw_mean=("min_ra_raw", "mean"),
            min_ra_raw_std=("min_ra_raw", "std"),
            min_ra_raw_min=("min_ra_raw", "min"),
            min_ra_raw_median=("min_ra_raw", "median"),
            min_ra_raw_max=("min_ra_raw", "max"),
        )
    )
    grouped_instance_group.to_csv(
        os.path.join(out_path, "hv_igd_summary_by_instance_group.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 6) summary: group overall
    grouped_group = (
        df_run.groupby(["group_label"], as_index=False)
        .agg(
            hv_unified_count=("hv_unified", "count"),
            hv_unified_mean=("hv_unified", "mean"),
            hv_unified_std=("hv_unified", "std"),
            hv_unified_min=("hv_unified", "min"),
            hv_unified_median=("hv_unified", "median"),
            hv_unified_max=("hv_unified", "max"),

            igd_plus_unified_count=("igd_plus_unified", "count"),
            igd_plus_unified_mean=("igd_plus_unified", "mean"),
            igd_plus_unified_std=("igd_plus_unified", "std"),
            igd_plus_unified_min=("igd_plus_unified", "min"),
            igd_plus_unified_median=("igd_plus_unified", "median"),
            igd_plus_unified_max=("igd_plus_unified", "max"),

            pareto_size_count=("pareto_size", "count"),
            pareto_size_mean=("pareto_size", "mean"),
            pareto_size_std=("pareto_size", "std"),
            pareto_size_min=("pareto_size", "min"),
            pareto_size_median=("pareto_size", "median"),
            pareto_size_max=("pareto_size", "max"),

            min_cost_pareto_count=("min_cost_pareto", "count"),
            min_cost_pareto_mean=("min_cost_pareto", "mean"),
            min_cost_pareto_std=("min_cost_pareto", "std"),
            min_cost_pareto_min=("min_cost_pareto", "min"),
            min_cost_pareto_median=("min_cost_pareto", "median"),
            min_cost_pareto_max=("min_cost_pareto", "max"),

            min_ra_pareto_count=("min_ra_pareto", "count"),
            min_ra_pareto_mean=("min_ra_pareto", "mean"),
            min_ra_pareto_std=("min_ra_pareto", "std"),
            min_ra_pareto_min=("min_ra_pareto", "min"),
            min_ra_pareto_median=("min_ra_pareto", "median"),
            min_ra_pareto_max=("min_ra_pareto", "max"),

            min_cost_raw_count=("min_cost_raw", "count"),
            min_cost_raw_mean=("min_cost_raw", "mean"),
            min_cost_raw_std=("min_cost_raw", "std"),
            min_cost_raw_min=("min_cost_raw", "min"),
            min_cost_raw_median=("min_cost_raw", "median"),
            min_cost_raw_max=("min_cost_raw", "max"),

            min_ra_raw_count=("min_ra_raw", "count"),
            min_ra_raw_mean=("min_ra_raw", "mean"),
            min_ra_raw_std=("min_ra_raw", "std"),
            min_ra_raw_min=("min_ra_raw", "min"),
            min_ra_raw_median=("min_ra_raw", "median"),
            min_ra_raw_max=("min_ra_raw", "max"),
        )
    )
    grouped_group.to_csv(
        os.path.join(out_path, "hv_igd_summary_by_group_overall.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(norm_info_rows).to_csv(
        os.path.join(out_path, "normalization_info.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(ref_front_rows).to_csv(
        os.path.join(out_path, "reference_front_by_instance.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    if skipped_files:
        pd.DataFrame(skipped_files, columns=["path_or_file", "reason"]).to_csv(
            os.path.join(out_path, "skipped_files.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    print("=" * 90)
    print("Done.")
    print(f"Input path:  {path}")
    print(f"Output path: {out_path}")
    print(f"Processed files: {len(df_run)}")
    print(f"Instances: {df_run['instance'].nunique()}")
    print(f"Groups: {sorted(df_run['group_label'].unique().tolist())}")
    print(f"Betas: {sorted(df_run['beta'].unique().tolist())}")
    print("=" * 90)



if __name__ == "__main__":
    path = r"D:\02_Research\6_experimentResults\Paper_GB_MOVNS\Ablation\without_GB\0427testGB\pareto"
    out_path = r"D:\02_Research\6_experimentResults\Paper_GB_MOVNS\Ablation\without_GB\0427testGB\result"

    main(path, out_path, folder_level=0)