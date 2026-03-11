import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import pandas as pd



from Experiment.ccp_adapter import _build_summary_ccp, _make_archive_rows_ccp, _make_key_ccp, _make_tasks_ccp, _run_one_task_ccp, _save_run_details_pkl_ccp
from Experiment.common import _run_parallel_common
from Experiment.spr_adapter import _build_summary_spr, _make_archive_rows_spr, _make_key_spr, _make_tasks_spr, _run_one_task_spr, _save_run_details_pkl_spr






def run_experiments_parallel_spr(
    BETAS,
    N_RUNS,
    instance_source: str,
    results_dir: Optional[str] = None,
    base_seed: int = 42,
    max_workers: int = 10,
    update_summary_each_run: bool = True,
    save_every: int = 10,
    continue_on_error: bool = True,
    create_timestamp_subdir: bool = False,
):
    def make_tasks(instance_list, done):
        return _make_tasks_spr(instance_list, BETAS, N_RUNS, base_seed, done)

    def save_pkl(details_dir, detail_rec, **out):
        return _save_run_details_pkl_spr(
            details_dir=details_dir,
            instance_name=out["instance_name"],
            beta=out["beta"],
            run_id=out["run_id"],
            detail_rec=detail_rec,
        )

    return _run_parallel_common(
        mode_name="spr",
        instance_source=instance_source,
        results_dir=results_dir,
        max_workers=max_workers,
        update_summary_each_run=update_summary_each_run,
        save_every=save_every,
        continue_on_error=continue_on_error,
        create_timestamp_subdir=create_timestamp_subdir,
        progress_key_columns=["instance", "beta", "run_id"],
        make_key=_make_key_spr,
        make_tasks=make_tasks,
        run_one_task=_run_one_task_spr,
        build_summary=_build_summary_spr,
        make_archive_rows=_make_archive_rows_spr,
        save_run_details_pkl=save_pkl,
    )


def run_experiments_parallel_ccp(
    BETAS,
    ALPHAS,
    N_RUNS,
    instance_source: str,
    results_dir: Optional[str] = None,
    base_seed: int = 42,
    max_workers: int = 10,
    update_summary_each_run: bool = True,
    save_every: int = 10,
    continue_on_error: bool = True,
    create_timestamp_subdir: bool = False,
):
    def make_tasks(instance_list, done):
        return _make_tasks_ccp(instance_list, BETAS, ALPHAS, N_RUNS, base_seed, done)

    def save_pkl(details_dir, detail_rec, **out):
        return _save_run_details_pkl_ccp(
            details_dir=details_dir,
            instance_name=out["instance_name"],
            beta=out["beta"],
            alpha=out["alpha"],
            run_id=out["run_id"],
            detail_rec=detail_rec,
        )

    return _run_parallel_common(
        mode_name="ccp",
        instance_source=instance_source,
        results_dir=results_dir,
        max_workers=max_workers,
        update_summary_each_run=update_summary_each_run,
        save_every=save_every,
        continue_on_error=continue_on_error,
        create_timestamp_subdir=create_timestamp_subdir,
        progress_key_columns=["instance", "beta", "alpha", "run_id"],
        make_key=_make_key_ccp,
        make_tasks=make_tasks,
        run_one_task=_run_one_task_ccp,
        build_summary=_build_summary_ccp,
        make_archive_rows=_make_archive_rows_ccp,
        save_run_details_pkl=save_pkl,
    )