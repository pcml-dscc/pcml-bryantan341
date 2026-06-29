# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 2: Hypothesis Testing, Bootstrap & CUPED

Complete the `solve()` function. Read problem.md for the full specification.
The bootstrap is auto-graded against a bit-reproducible reference: you MUST
follow the seed / resample protocol exactly (treatment resampled before
control, every iteration).

    python grader.py starter.py
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from shared import MLFPDataLoader

# --- Fixed problem constants (do not change) ---
COHORT = ["control", "treatment_a"]
BOOT_SEED = 2024            # np.random.default_rng(BOOT_SEED)
BOOT_B = 2000               # number of bootstrap resamples
MT_P_VALUES = [0.03, 0.012, 0.04, 0.65, 0.009]   # five simultaneous tests
MT_ALPHA = 0.05


def solve() -> dict:
    """Return the hypothesis-testing / bootstrap / CUPED answer dict.

    See problem.md for the exact 13 keys and how each is defined.
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp02", "experiment_data.parquet")
    co = df.filter(pl.col("experiment_group").is_in(COHORT))

    # TODO 1: Extract metric_value for treatment_a (`t`) and control (`c`) as
    #         float numpy arrays, in file order. Welch two-sample t-test:
    #         welch_t, welch_p = stats.ttest_ind(t, c, equal_var=False).
    #         mean_diff = t.mean() - c.mean().
    t = (
        co
        .filter(pl.col("experiment_group") == "treatment_a")
        .select(pl.col("metric_value").cast(pl.Float64))
        .to_series()
        .to_numpy()
    )

    c = (
        co
        .filter(pl.col("experiment_group") == "control")
        .select(pl.col("metric_value").cast(pl.Float64))
        .to_series()
        .to_numpy()
    )

    welch_t, welch_p = stats.ttest_ind(
        t,
        c,
        equal_var=False
    )

    mean_diff = t.mean() - c.mean()

    # TODO 2: Seeded percentile bootstrap of mean_diff (EXACT protocol):
    #             rng = np.random.default_rng(BOOT_SEED)
    #             for b in range(BOOT_B):
    #                 bt = rng.choice(t, size=t.size, replace=True)   # treatment FIRST
    #                 bc = rng.choice(c, size=c.size, replace=True)   # control SECOND
    #                 diffs[b] = bt.mean() - bc.mean()
    #             boot_ci_low, boot_ci_high = np.percentile(diffs, [2.5, 97.5])
    rng = np.random.default_rng(BOOT_SEED)
    diffs = np.empty(BOOT_B)

    for b in range(BOOT_B):
        bt = rng.choice(
            t,
            size=t.size,
            replace=True
        )

        bc = rng.choice(
            c,
            size=c.size,
            replace=True
        )

        diffs[b] = bt.mean() - bc.mean()

    boot_ci_low, boot_ci_high = np.percentile(
        diffs,
        [2.5, 97.5]
    )

    # TODO 3: CUPED with pre_metric_value as covariate, over the full cohort:
    #             theta = np.cov(metric, pre, ddof=1)[0,1] / np.var(pre, ddof=1)
    #             metric_adj = metric - theta * (pre - pre.mean())
    #             var_metric = np.var(metric, ddof=1); var_adj = np.var(metric_adj, ddof=1)
    #             cuped_var_reduction = 1 - var_adj / var_metric
    metric = (
        co
        .select(pl.col("metric_value").cast(pl.Float64))
        .to_series()
        .to_numpy()
    )

    pre = (
        co
        .select(pl.col("pre_metric_value").cast(pl.Float64))
        .to_series()
        .to_numpy()
    )

    cuped_theta = (
        np.cov(metric, pre, ddof=1)[0, 1]
        / np.var(pre, ddof=1)
    )

    metric_adj = (
        metric
        - cuped_theta * (pre - pre.mean())
    )

    var_metric = np.var(metric, ddof=1)
    var_adj = np.var(metric_adj, ddof=1)

    cuped_var_reduction = (
        1.0
        - var_adj / var_metric
    )

    # TODO 4: Re-run the Welch test on metric_adj (treatment vs control) ->
    #         welch_t_cuped, welch_p_cuped.
    groups = (
        co
        .select("experiment_group")
        .to_series()
        .to_numpy()
    )

    t_adj = metric_adj[groups == "treatment_a"]
    c_adj = metric_adj[groups == "control"]

    welch_t_cuped, welch_p_cuped = stats.ttest_ind(
        t_adj,
        c_adj,
        equal_var=False
    )

    # TODO 5: Multiple testing over MT_P_VALUES at MT_ALPHA:
    #           - Bonferroni: count pv < alpha/m  -> bonferroni_n_sig
    #           - Benjamini-Hochberg step-up: sort p, threshold_i = alpha*i/m,
    #             reject all up to the largest i with p_(i) <= threshold_i
    #             -> bh_n_sig
    p_values = np.asarray(
        MT_P_VALUES,
        dtype=float
    )

    m = p_values.size

    bonferroni_threshold = MT_ALPHA / m

    bonferroni_n_sig = int(
        np.sum(p_values < bonferroni_threshold)
    )

    sorted_p_values = np.sort(p_values)

    bh_thresholds = (
        MT_ALPHA
        * np.arange(1, m + 1)
        / m
    )

    valid_ranks = np.where(
        sorted_p_values <= bh_thresholds
    )[0]

    if valid_ranks.size == 0:
        bh_n_sig = 0
    else:
        bh_n_sig = int(valid_ranks[-1] + 1)

    # TODO 6: Return the dict with all 13 keys (see problem.md).
    return {
        "welch_t": float(welch_t),
        "welch_p": float(welch_p),
        "mean_diff": float(mean_diff),
        "boot_ci_low": float(boot_ci_low),
        "boot_ci_high": float(boot_ci_high),
        "cuped_theta": float(cuped_theta),
        "var_metric": float(var_metric),
        "var_adj": float(var_adj),
        "cuped_var_reduction": float(cuped_var_reduction),
        "welch_t_cuped": float(welch_t_cuped),
        "welch_p_cuped": float(welch_p_cuped),
        "bonferroni_n_sig": bonferroni_n_sig,
        "bh_n_sig": bh_n_sig,
    }


if __name__ == "__main__":
    print(solve())