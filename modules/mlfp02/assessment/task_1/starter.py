# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 1: Probability, Bayes & Experiment Validation

Complete the `solve()` function. Read problem.md for the full specification.

    python grader.py starter.py
"""
from __future__ import annotations

import polars as pl
from scipy import stats

from shared import MLFPDataLoader

# --- Fixed problem constants (do not change) ---
COHORT = ["control", "treatment_a"]
CONVERT_THRESHOLD = 50.0
FRAUD_BASE_RATE = 0.02
FRAUD_SENSITIVITY = 0.95
FRAUD_FPR = 0.03
BETA_PRIOR_ALPHA = 2.0
BETA_PRIOR_BETA = 20.0


def solve() -> dict:
    """Return the probability / Bayes / experiment-validation answer dict."""

    loader = MLFPDataLoader()
    df = loader.load("mlfp02", "experiment_data.parquet")

    # ============================================================
    # TASK 1:
    # Keep only control and treatment_a.
    # Create converted = True when metric_value is at least 50.
    # ============================================================
    cohort_df = (
        df
        .filter(pl.col("experiment_group").is_in(COHORT))
        .with_columns(
            (pl.col("metric_value") >= CONVERT_THRESHOLD)
            .alias("converted")
        )
    )

    # Get total rows and total conversions in the cohort.
    cohort_summary = (
        cohort_df
        .select(
            pl.len().alias("n_total"),
            pl.col("converted").sum().alias("n_converted"),
        )
        .row(0, named=True)
    )

    n_total = int(cohort_summary["n_total"])
    n_converted = int(cohort_summary["n_converted"])

    # Get control group count and conversions.
    control_summary = (
        cohort_df
        .filter(pl.col("experiment_group") == "control")
        .select(
            pl.len().alias("n_control"),
            pl.col("converted").sum().alias("control_successes"),
        )
        .row(0, named=True)
    )

    n_control = int(control_summary["n_control"])
    control_successes = int(control_summary["control_successes"])

    # Get treatment_a count and conversions.
    treatment_summary = (
        cohort_df
        .filter(pl.col("experiment_group") == "treatment_a")
        .select(
            pl.len().alias("n_treatment"),
            pl.col("converted").sum().alias("treatment_successes"),
        )
        .row(0, named=True)
    )

    n_treatment = int(treatment_summary["n_treatment"])
    treatment_successes = int(
        treatment_summary["treatment_successes"]
    )

    # ============================================================
    # TASK 2:
    # Calculate the three required conversion probabilities.
    # ============================================================
    p_convert_overall = n_converted / n_total

    p_convert_control = (
        control_successes / n_control
    )

    p_convert_treatment = (
        treatment_successes / n_treatment
    )

    # ============================================================
    # TASK 3:
    # Bayes inversion:
    # P(treatment_a | converted)
    # ============================================================
    p_treatment = n_treatment / n_total

    p_treatment_given_convert = (
        p_convert_treatment
        * p_treatment
        / p_convert_overall
    )

    # ============================================================
    # TASK 4:
    # Sample Ratio Mismatch test against a 50/50 split.
    # ============================================================
    expected_count = n_total / 2.0

    srm_chi2 = (
        ((n_control - expected_count) ** 2 / expected_count)
        + ((n_treatment - expected_count) ** 2 / expected_count)
    )

    srm_p_value = stats.chi2.sf(
        srm_chi2,
        df=1,
    )

    srm_flag = srm_p_value < 1e-3

    # ============================================================
    # TASK 5:
    # Fraud detector base-rate Bayes calculation.
    # ============================================================
    fraud_numerator = (
        FRAUD_SENSITIVITY
        * FRAUD_BASE_RATE
    )

    fraud_denominator = (
        fraud_numerator
        + FRAUD_FPR * (1.0 - FRAUD_BASE_RATE)
    )

    p_fraud_given_flagged = (
        fraud_numerator
        / fraud_denominator
    )

    # ============================================================
    # TASK 6:
    # Beta-Binomial posterior update for treatment_a.
    # ============================================================
    treatment_failures = (
        n_treatment
        - treatment_successes
    )

    beta_post_alpha = (
        BETA_PRIOR_ALPHA
        + treatment_successes
    )

    beta_post_beta = (
        BETA_PRIOR_BETA
        + treatment_failures
    )

    posterior_mean = (
        beta_post_alpha
        / (beta_post_alpha + beta_post_beta)
    )

    cred_int_low = stats.beta.ppf(
        0.025,
        beta_post_alpha,
        beta_post_beta,
    )

    cred_int_high = stats.beta.ppf(
        0.975,
        beta_post_alpha,
        beta_post_beta,
    )

    # ============================================================
    # TASK 7:
    # Return all 13 required values using the exact key names.
    # ============================================================
    return {
        "p_convert_overall": float(p_convert_overall),
        "p_convert_control": float(p_convert_control),
        "p_convert_treatment": float(p_convert_treatment),
        "p_treatment_given_convert": float(
            p_treatment_given_convert
        ),
        "srm_chi2": float(srm_chi2),
        "srm_p_value": float(srm_p_value),
        "srm_flag": bool(srm_flag),
        "p_fraud_given_flagged": float(
            p_fraud_given_flagged
        ),
        "beta_post_alpha": float(beta_post_alpha),
        "beta_post_beta": float(beta_post_beta),
        "posterior_mean": float(posterior_mean),
        "cred_int_low": float(cred_int_low),
        "cred_int_high": float(cred_int_high),
    }


if __name__ == "__main__":
    print(solve())