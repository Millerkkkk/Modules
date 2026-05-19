import os
import re
import json
import copy
from collections import defaultdict

import numpy as np
import pandas as pd


def dominates_min(a, b, eps=1e-12):
    """
    Bi-objective minimization domination:
    a dominates b iff
    a_i <= b_i for all i, and strictly < in at least one i.
    """
    return (
        a[0] <= b[0] + eps
        and a[1] <= b[1] + eps
        and (a[0] < b[0] - eps or a[1] < b[1] - eps)
    )


def nondominated_filter(points, eps=1e-12):
    """
    Keep only nondominated points for bi-objective minimization.
    Deduplicate first.
    """
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
    """
    2D hypervolume for minimization with a fixed reference point.
    """
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
    """
    Instance-wise min-max normalization.
    """
    dc = max(cmax - cmin, eps)
    dr = max(rmax - rmin, eps)

    normed = []
    for c, r in points:
        cn = (float(c) - cmin) / dc
        rn = (float(r) - rmin) / dr
        normed.append((cn, rn))
    return normed


def igd_plus(reference_front, approx_front):
    """
    IGD+ for minimization.

    For each reference point z in reference_front:
        d^+(z, A) = min_{a in A} sqrt(sum_i max(a_i - z_i, 0)^2)

    IGD+(A, R) = average over z in R of d^+(z, A)

    Smaller is better.
    """
    if not reference_front:
        return np.nan

    if not approx_front:
        return np.inf

    ref_arr = np.asarray(reference_front, dtype=float)   # shape (m, 2)
    app_arr = np.asarray(approx_front, dtype=float)      # shape (n, 2)

    distances = []
    for z in ref_arr:
        diff = np.maximum(app_arr - z, 0.0)
        d = np.sqrt(np.sum(diff * diff, axis=1))
        distances.append(np.min(d))

    return float(np.mean(distances))


def load_points_from_json(json_path):
    """
    Priority:
    1) pareto_points
    2) archive -> [(cost, ra), ...]
    """
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


def main(path, out_path):
    ensure_dir(out_path)
    out_json_dir = os.path.join(out_path, "json_with_unified_metrics")
    ensure_dir(out_json_dir)

    # 文件名示例: r108_21_beta0.100_run7.json
    pattern = re.compile(
        r"^(?P<instance>.+?)_beta(?P<beta>\d+(?:\.\d+)?)_run(?P<run>\d+)\.json$",
        re.IGNORECASE,
    )

    files_by_instance = defaultdict(list)
    skipped_files = []

    # 1) 扫描文件
    for fn in os.listdir(path):
        if not fn.lower().endswith(".json"):
            continue

        m = pattern.match(fn)
        if not m:
            skipped_files.append((fn, "filename_not_matched"))
            continue

        instance = m.group("instance")
        beta = float(m.group("beta"))
        run = int(m.group("run"))
        full_path = os.path.join(path, fn)

        files_by_instance[instance].append(
            {
                "filename": fn,
                "path": full_path,
                "instance": instance,
                "beta": beta,
                "run": run,
            }
        )

    if not files_by_instance:
        raise RuntimeError("No valid json files found. Please check path and filename pattern.")

    per_run_rows = []
    norm_info_rows = []
    ref_front_rows = []

    # 2) 按 instance 统一处理
    for instance, items in sorted(files_by_instance.items()):
        union_points = []

        # 2.1 读取每个文件的 Pareto 点
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
            skipped_files.append((bad["filename"], bad.get("load_error", "unknown_error")))

        if not valid_items:
            continue

        # 2.2 计算 instance-wise normalization range
        all_costs = [float(c) for c, _ in union_points]
        all_ras = [float(r) for _, r in union_points]

        cmin = min(all_costs)
        cmax = max(all_costs)
        rmin = min(all_ras)
        rmax = max(all_ras)

        # 2.3 构造 instance 的 unified reference front
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

        # 2.4 对每个 run 重算 unified HV 和 IGD+
        for item in valid_items:
            raw_points = item["points_raw"]
            nd_raw = nondominated_filter(raw_points)

            norm_points = normalize_points(nd_raw, cmin, cmax, rmin, rmax)
            nd_norm = nondominated_filter(norm_points)
            nd_norm = sorted(nd_norm, key=lambda x: (x[0], x[1]))

            hv_unified = hypervolume_2d_min(nd_norm, ref_point=(1.1, 1.1))
            igd_plus_unified = igd_plus(reference_front, nd_norm)

            row = {
                "instance": item["instance"],
                "beta": item["beta"],
                "run": item["run"],
                "filename": item["filename"],
                "n_points_raw": len(raw_points),
                "n_points_nd_raw": len(nd_raw),
                "n_points_nd_norm": len(nd_norm),
                "n_reference_front_points": len(reference_front),
                "hv_unified": float(hv_unified),
                "igd_plus_unified": float(igd_plus_unified),
                "cost_min_instance": float(cmin),
                "cost_max_instance": float(cmax),
                "ra_min_instance": float(rmin),
                "ra_max_instance": float(rmax),
            }
            per_run_rows.append(row)

            # 另存带统一指标的 json
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

            out_json_path = os.path.join(out_json_dir, item["filename"])
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)

    # 3) 输出 per-run 表
    df_run = pd.DataFrame(per_run_rows)
    if df_run.empty:
        raise RuntimeError("No valid results were processed.")

    df_run = df_run.sort_values(by=["instance", "beta", "run"]).reset_index(drop=True)
    df_run.to_csv(
        os.path.join(out_path, "hv_igd_per_run.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 4) 汇总到 instance-beta
    grouped = (
        df_run.groupby(["instance", "beta"], as_index=False)
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

            n_points_nd_norm_mean=("n_points_nd_norm", "mean"),
            n_reference_front_points=("n_reference_front_points", "mean"),
        )
    )

    grouped = grouped.sort_values(by=["instance", "beta"]).reset_index(drop=True)
    grouped.to_csv(
        os.path.join(out_path, "hv_igd_summary_by_instance_beta.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 5) 再汇总到 beta overall
    grouped_beta = (
        df_run.groupby(["beta"], as_index=False)
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
        )
    )

    grouped_beta = grouped_beta.sort_values(by=["beta"]).reset_index(drop=True)
    grouped_beta.to_csv(
        os.path.join(out_path, "hv_igd_summary_by_beta_overall.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 6) 保存 normalization info
    df_norm = pd.DataFrame(norm_info_rows)
    df_norm = df_norm.sort_values(by=["instance"]).reset_index(drop=True)
    df_norm.to_csv(
        os.path.join(out_path, "normalization_info.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 7) 保存 reference front
    df_ref = pd.DataFrame(ref_front_rows)
    df_ref = df_ref.sort_values(by=["instance", "ref_point_id"]).reset_index(drop=True)
    df_ref.to_csv(
        os.path.join(out_path, "reference_front_by_instance.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 8) 保存 skipped files
    if skipped_files:
        df_skip = pd.DataFrame(skipped_files, columns=["filename", "reason"])
        df_skip.to_csv(
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
    print(f"Betas: {sorted(df_run['beta'].unique().tolist())}")
    if skipped_files:
        print(f"Skipped files: {len(skipped_files)}")
        print("See skipped_files.csv for details.")
    print("=" * 90)




if __name__ == "__main__":
    path = r"D:\02_Research\6_experimentResults\Paper_GB_MOVNS\Sensitivity\Beta\20260412_235537\run_details_json"
    out_path = r"D:\02_Research\6_experimentResults\Paper_GB_MOVNS\Sensitivity\Beta\20260412_235537"

    main(path, out_path)


    