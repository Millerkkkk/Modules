import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def make_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_iter_log(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = [
        "instance", "beta", "run_id", "seed", "iter",
        "elapsed_sec", "eval_count",
        "best_cost_in_archive", "best_ra_in_archive"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"iter_log.csv 缺少这些列: {missing}")

    # 如果 hv 没有，也可以继续画 cost / ra
    if "hv" not in df.columns:
        print("[Warn] 列 'hv' 不存在，将只画 cost / ra。")

    df = df.sort_values(["instance", "beta", "run_id", "seed", "iter"]).reset_index(drop=True)
    return df


def smooth_series(y, window=1):
    if window <= 1:
        return y
    return pd.Series(y).rolling(window=window, min_periods=1).mean().values


def plot_group_convergence(
    gdf: pd.DataFrame,
    out_dir: str,
    instance: str,
    beta: float,
    smooth_window: int = 1
):
    """
    对某个 instance + beta，画三类图：
    1) cost vs elapsed_sec
    2) ra vs elapsed_sec
    3) hv vs elapsed_sec（如果有）
    4) cost vs eval_count
    5) ra vs eval_count
    6) hv vs eval_count（如果有）
    """

    runs = sorted(gdf[["run_id", "seed"]].drop_duplicates().itertuples(index=False, name=None))

    # -------- 1. 时间-收敛图 --------
    fig, axes = plt.subplots(3 if "hv" in gdf.columns else 2, 1, figsize=(8, 12), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax_cost = axes[0]
    ax_ra = axes[1]
    ax_hv = axes[2] if "hv" in gdf.columns else None

    for run_id, seed in runs:
        sdf = gdf[(gdf["run_id"] == run_id) & (gdf["seed"] == seed)].sort_values("elapsed_sec")

        x_time = sdf["elapsed_sec"].values
        y_cost = smooth_series(sdf["best_cost_in_archive"].values, window=smooth_window)
        y_ra = smooth_series(sdf["best_ra_in_archive"].values, window=smooth_window)

        ax_cost.plot(x_time, y_cost, marker="o", linewidth=1.5, label=f"run{run_id}-seed{seed}")
        ax_ra.plot(x_time, y_ra, marker="o", linewidth=1.5, label=f"run{run_id}-seed{seed}")

        if ax_hv is not None:
            y_hv = smooth_series(sdf["hv"].values, window=smooth_window)
            ax_hv.plot(x_time, y_hv, marker="o", linewidth=1.5, label=f"run{run_id}-seed{seed}")

    ax_cost.set_title(f"{instance} | beta={beta} | Best Cost vs Time")
    ax_cost.set_xlabel("Elapsed Time (sec)")
    ax_cost.set_ylabel("Best Cost in Archive")
    ax_cost.grid(True, alpha=0.3)
    ax_cost.legend(fontsize=8)

    ax_ra.set_title(f"{instance} | beta={beta} | Best RA vs Time")
    ax_ra.set_xlabel("Elapsed Time (sec)")
    ax_ra.set_ylabel("Best RA in Archive")
    ax_ra.grid(True, alpha=0.3)
    ax_ra.legend(fontsize=8)

    if ax_hv is not None:
        ax_hv.set_title(f"{instance} | beta={beta} | HV vs Time")
        ax_hv.set_xlabel("Elapsed Time (sec)")
        ax_hv.set_ylabel("HV")
        ax_hv.grid(True, alpha=0.3)
        ax_hv.legend(fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{instance}_beta{beta:.3f}_convergence_by_time.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    # -------- 2. 评估次数-收敛图 --------
    fig, axes = plt.subplots(3 if "hv" in gdf.columns else 2, 1, figsize=(8, 12), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax_cost = axes[0]
    ax_ra = axes[1]
    ax_hv = axes[2] if "hv" in gdf.columns else None

    for run_id, seed in runs:
        sdf = gdf[(gdf["run_id"] == run_id) & (gdf["seed"] == seed)].sort_values("eval_count")

        x_eval = sdf["eval_count"].values
        y_cost = smooth_series(sdf["best_cost_in_archive"].values, window=smooth_window)
        y_ra = smooth_series(sdf["best_ra_in_archive"].values, window=smooth_window)

        ax_cost.plot(x_eval, y_cost, marker="o", linewidth=1.5, label=f"run{run_id}-seed{seed}")
        ax_ra.plot(x_eval, y_ra, marker="o", linewidth=1.5, label=f"run{run_id}-seed{seed}")

        if ax_hv is not None:
            y_hv = smooth_series(sdf["hv"].values, window=smooth_window)
            ax_hv.plot(x_eval, y_hv, marker="o", linewidth=1.5, label=f"run{run_id}-seed{seed}")

    ax_cost.set_title(f"{instance} | beta={beta} | Best Cost vs Eval Count")
    ax_cost.set_xlabel("Evaluation Count")
    ax_cost.set_ylabel("Best Cost in Archive")
    ax_cost.grid(True, alpha=0.3)
    ax_cost.legend(fontsize=8)

    ax_ra.set_title(f"{instance} | beta={beta} | Best RA vs Eval Count")
    ax_ra.set_xlabel("Evaluation Count")
    ax_ra.set_ylabel("Best RA in Archive")
    ax_ra.grid(True, alpha=0.3)
    ax_ra.legend(fontsize=8)

    if ax_hv is not None:
        ax_hv.set_title(f"{instance} | beta={beta} | HV vs Eval Count")
        ax_hv.set_xlabel("Evaluation Count")
        ax_hv.set_ylabel("HV")
        ax_hv.grid(True, alpha=0.3)
        ax_hv.legend(fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{instance}_beta{beta:.3f}_convergence_by_eval.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def build_final_iter_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    每个 instance + beta + run_id + seed 只保留最后一个 iter，
    用来后续比较不同 seed 的最终结果。
    """
    idx = df.groupby(["instance", "beta", "run_id", "seed"])["iter"].idxmax()
    final_df = df.loc[idx].sort_values(["instance", "beta", "run_id", "seed"]).reset_index(drop=True)
    return final_df


def summarize_final_table(final_df: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "elapsed_sec": ["mean", "std", "max"],
        "eval_count": ["mean", "std", "max"],
        "best_cost_in_archive": ["mean", "std", "min"],
        "best_ra_in_archive": ["mean", "std", "min"],
    }
    if "hv" in final_df.columns:
        agg["hv"] = ["mean", "std", "max"]

    summary = final_df.groupby(["instance", "beta"]).agg(agg)
    summary.columns = ["_".join(col) for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    return summary


def main():
    csv_path = r"D:\02_Research\6_experimentResults\Paper_GB_MOVNS\Ablation\para_max_time\20260420_233436\iter_log.csv"      # 改成你的路径
    out_dir = r"D:\02_Research\6_experimentResults\Paper_GB_MOVNS\Ablation\para_max_time\20260420_233436\plot"  # 图片输出文件夹
    smooth_window = 1               # 不平滑就设 1；想稍微平滑可设 2 或 3

    make_output_dir(out_dir)
    df = load_iter_log(csv_path)

    # 逐个 instance + beta 画图
    groups = df.groupby(["instance", "beta"])
    for (instance, beta), gdf in groups:
        print(f"[Plot] {instance} | beta={beta}")
        plot_group_convergence(
            gdf=gdf,
            out_dir=out_dir,
            instance=instance,
            beta=beta,
            smooth_window=smooth_window,
        )

    # 导出最终迭代表和汇总表
    final_df = build_final_iter_table(df)
    summary_df = summarize_final_table(final_df)

    final_df.to_csv(os.path.join(out_dir, "final_iter_per_run.csv"), index=False, encoding="utf-8-sig")
    summary_df.to_csv(os.path.join(out_dir, "summary_final_per_instance_beta.csv"), index=False, encoding="utf-8-sig")

    print("[Done] 图和汇总表已保存到:", out_dir)


if __name__ == "__main__":
    main()