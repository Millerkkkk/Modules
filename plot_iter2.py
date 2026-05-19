import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_iter_log(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = [
        "instance", "beta", "run_id", "seed", "iter",
        "elapsed_sec", "eval_count",
        "best_cost_in_archive", "best_ra_in_archive"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if "hv" not in df.columns:
        print("[Warn] hv column not found. HV plot will be skipped.")

    df = df.sort_values(["instance", "beta", "run_id", "seed", "elapsed_sec"]).reset_index(drop=True)
    return df


def align_run_to_time_grid(run_df: pd.DataFrame, time_grid: np.ndarray, metrics: list[str]) -> pd.DataFrame:
    """
    对单个 run 按统一时间网格对齐。
    采用 '截至该时刻最近一次观测值' 的方式（forward fill / step-wise）。
    """
    run_df = run_df.sort_values("elapsed_sec").copy()

    base = pd.DataFrame({"elapsed_sec": time_grid})
    merged = pd.merge_asof(
        base,
        run_df[["elapsed_sec"] + metrics],
        on="elapsed_sec",
        direction="backward"
    )

    # 对于第一个记录点之前的时间，继续用第一条记录填充，避免前面是 NaN
    for col in metrics:
        if merged[col].isna().all():
            continue
        merged[col] = merged[col].bfill()

    return merged


def build_time_aligned_summary(
    df: pd.DataFrame,
    instance: str,
    beta: float,
    time_step: int = 200,
    max_time: int | None = None,
):
    """
    对某个 instance + beta，把所有 run 对齐到统一时间网格，并计算 mean/std。
    """
    gdf = df[(df["instance"] == instance) & (df["beta"] == beta)].copy()
    if gdf.empty:
        return None, None

    if max_time is None:
        max_time = int(np.ceil(gdf["elapsed_sec"].max() / time_step) * time_step)

    time_grid = np.arange(0, max_time + time_step, time_step)

    metrics = ["best_cost_in_archive", "best_ra_in_archive"]
    if "hv" in gdf.columns:
        metrics.append("hv")

    aligned_runs = []
    run_keys = sorted(gdf[["run_id", "seed"]].drop_duplicates().itertuples(index=False, name=None))

    for run_id, seed in run_keys:
        rdf = gdf[(gdf["run_id"] == run_id) & (gdf["seed"] == seed)].copy()
        aligned = align_run_to_time_grid(rdf, time_grid, metrics)
        aligned["run_id"] = run_id
        aligned["seed"] = seed
        aligned_runs.append(aligned)

    aligned_df = pd.concat(aligned_runs, ignore_index=True)

    agg_dict = {}
    for m in metrics:
        agg_dict[m] = ["mean", "std", "median", "min", "max"]

    summary = aligned_df.groupby("elapsed_sec").agg(agg_dict)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()

    return aligned_df, summary


def plot_time_aligned_mean_curve(summary: pd.DataFrame, out_path: str, title_prefix: str):
    has_hv = "hv_mean" in summary.columns
    nrows = 3 if has_hv else 2

    fig, axes = plt.subplots(nrows, 1, figsize=(8, 11))
    if nrows == 1:
        axes = [axes]

    x = summary["elapsed_sec"].values

    # cost
    ax = axes[0]
    y = summary["best_cost_in_archive_mean"].values
    s = summary["best_cost_in_archive_std"].fillna(0).values
    ax.plot(x, y, linewidth=2, label="Mean")
    ax.fill_between(x, y - s, y + s, alpha=0.25, label="Mean ± Std")
    ax.set_title(f"{title_prefix} | Best Cost vs Time")
    ax.set_xlabel("Elapsed Time (sec)")
    ax.set_ylabel("Best Cost")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ra
    ax = axes[1]
    y = summary["best_ra_in_archive_mean"].values
    s = summary["best_ra_in_archive_std"].fillna(0).values
    ax.plot(x, y, linewidth=2, label="Mean")
    ax.fill_between(x, y - s, y + s, alpha=0.25, label="Mean ± Std")
    ax.set_title(f"{title_prefix} | Best RA vs Time")
    ax.set_xlabel("Elapsed Time (sec)")
    ax.set_ylabel("Best RA")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # hv
    if has_hv:
        ax = axes[2]
        y = summary["hv_mean"].values
        s = summary["hv_std"].fillna(0).values
        ax.plot(x, y, linewidth=2, label="Mean")
        ax.fill_between(x, y - s, y + s, alpha=0.25, label="Mean ± Std")
        ax.set_title(f"{title_prefix} | HV vs Time")
        ax.set_xlabel("Elapsed Time (sec)")
        ax.set_ylabel("HV")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    csv_path = r"D:\02_Research\6_experimentResults\Paper_GB_MOVNS\Ablation\para_max_time\20260420_233436\iter_log.csv"      # 改成你的路径
    out_dir = r"D:\02_Research\6_experimentResults\Paper_GB_MOVNS\Ablation\para_max_time\20260420_233436\plot"  # 图片输出文件夹
    os.makedirs(out_dir, exist_ok=True)

    df = load_iter_log(csv_path)

    for (instance, beta), _ in df.groupby(["instance", "beta"]):
        aligned_df, summary = build_time_aligned_summary(
            df=df,
            instance=instance,
            beta=beta,
            time_step=200,   # 每 200 秒取一个点
            max_time=None
        )

        if summary is None:
            continue

        summary.to_csv(
            os.path.join(out_dir, f"{instance}_beta{beta:.3f}_time_aligned_summary.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        aligned_df.to_csv(
            os.path.join(out_dir, f"{instance}_beta{beta:.3f}_time_aligned_runs.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        plot_time_aligned_mean_curve(
            summary=summary,
            out_path=os.path.join(out_dir, f"{instance}_beta{beta:.3f}_mean_curve_by_time.png"),
            title_prefix=f"{instance} | beta={beta}"
        )

        print(f"[Done] {instance} | beta={beta}")

    print("[All Done]")


if __name__ == "__main__":
    main()