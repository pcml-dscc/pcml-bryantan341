# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 1: Probability, Bayes & Experiment Validation

Complete the `solve()` function. Read problem.md for the full specification.
Your submission is auto-graded: every probability, the SRM chi-square, the
base-rate Bayes scalar, and the Beta posterior must match the independently
re-derived reference within tight tolerances.

    python grader.py starter.py
"""
from __future__ import annotations

import polars as pl
from scipy import stats

from shared import MLFPDataLoader

# --- Fixed problem constants (do not change) ---
COHORT = ["control", "treatment_a"]
CONVERT_THRESHOLD = 50.0          # converted := metric_value >= 50.0
FRAUD_BASE_RATE = 0.02            # P(fraud)
FRAUD_SENSITIVITY = 0.95          # P(flagged | fraud)
FRAUD_FPR = 0.03                  # P(flagged | not fraud)
BETA_PRIOR_ALPHA = 2.0
BETA_PRIOR_BETA = 20.0


def solve() -> dict:
    """Return the probability / Bayes / experiment-validation answer dict.

    See problem.md for the exact 13 keys and how each is defined.
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp02", "experiment_data.parquet")

    # TODO 1: Restrict to the control + treatment_a cohort and add a boolean
    #         `converted` column = (metric_value >= CONVERT_THRESHOLD).
    cohort_df = (
        df
        .filter(pl.col("experiment_group").is_in(COHORT))
        .with_columns(
            (pl.col("metric_value") >= CONVERT_THRESHOLD).alias("converted")
        )
    )

    # Get the total number of rows and conversions in the cohort.
    cohort_summary = cohort_df.select(
        pl.len().alias("n_total"),
        pl.col("converted").sum().alias("n_converted")
    ).row(0, named=True)

    n_total = int(cohort_summary["n_total"])
    n_converted = int(cohort_summary["n_converted"])

    # Get the counts and conversions for the control arm.
    control_summary = (
        cohort_df
        .filter(pl.col("experiment_group") == "control")
        .select(
            pl.len().alias("n"),
            pl.col("converted").sum().alias("successes")
        )
        .row(0, named=True)
    )

    n_control = int(control_summary["n"])
    control_successes = int(control_summary["successes"])

    # Get the counts and conversions for the treatment_a arm.
    treatment_summary = (
        cohort_df
        .filter(pl.col("experiment_group") == "treatment_a")
        .select(
            pl.len().alias("n"),
            pl.col("converted").sum().alias("successes")
        )
        .row(0, named=True)
    )

    n_treatment = int(treatment_summary["n"])
    treatment_successes = int(treatment_summary["successes"])

    # TODO 2: Compute p_convert_overall, p_convert_control, p_convert_treatment.
    p_convert_overall = n_converted / n_total
    p_convert_control = control_successes / n_control
    p_convert_treatment = treatment_successes / n_treatment

    # TODO 3: Bayes inversion p_treatment_given_convert =
    #         P(converted|treatment) * P(treatment) / P(converted), where
    #         P(treatment) is the treatment_a share of the cohort.
    p_treatment = n_treatment / n_total

    p_treatment_given_convert = (
        p_convert_treatment
        * p_treatment
        / p_convert_overall
    )

    # TODO 4: SRM check vs a designed 50/50 split — chi-square goodness-of-fit
    #         on [n_control, n_treatment] with expected = n_total/2 each
    #         (df=1). srm_p_value = stats.chi2.sf(chi2, df=1);
    #         srm_flag = (srm_p_value < 1e-3).
    expected_count = n_total / 2.0

    srm_chi2 = (
        ((n_control - expected_count) ** 2 / expected_count)
        + ((n_treatment - expected_count) ** 2 / expected_count)
    )

    srm_p_value = float(stats.chi2.sf(srm_chi2, df=1))
    srm_flag = bool(srm_p_value < 1e-3)

    # TODO 5: Fraud base-rate Bayes (use the FRAUD_* constants):
    #         P(fraud|flagged) = sens*base / (sens*base + fpr*(1-base)).
    fraud_numerator = FRAUD_SENSITIVITY * FRAUD_BASE_RATE

    fraud_denominator = (
        fraud_numerator
        + FRAUD_FPR * (1.0 - FRAUD_BASE_RATE)
    )

    p_fraud_given_flagged = fraud_numerator / fraud_denominator

    # TODO 6: Beta-Binomial update on treatment_a: successes = sum(converted),
    #         failures = n - successes; posterior = Beta(prior_a+successes,
    #         prior_b+failures); posterior_mean = a/(a+b);
    #         95% credible interval via stats.beta.ppf(0.025/0.975, a, b).
    treatment_failures = n_treatment - treatment_successes

    beta_post_alpha = BETA_PRIOR_ALPHA + treatment_successes
    beta_post_beta = BETA_PRIOR_BETA + treatment_failures

    posterior_mean = (
        beta_post_alpha
        / (beta_post_alpha + beta_post_beta)
    )

    cred_int_low = float(
        stats.beta.ppf(
            0.025,
            beta_post_alpha,
            beta_post_beta
        )
    )

    cred_int_high = float(
        stats.beta.ppf(
            0.975,
            beta_post_alpha,
            beta_post_beta
        )
    )

    # TODO 7: Return the dict with all 13 keys (see problem.md).
    return {
        "p_convert_overall": float(p_convert_overall),
        "p_convert_control": float(p_convert_control),
        "p_convert_treatment": float(p_convert_treatment),
        "p_treatment_given_convert": float(p_treatment_given_convert),
        "srm_chi2": float(srm_chi2),
        "srm_p_value": float(srm_p_value),
        "srm_flag": srm_flag,
        "p_fraud_given_flagged": float(p_fraud_given_flagged),
        "beta_post_alpha": float(beta_post_alpha),
        "beta_post_beta": float(beta_post_beta),
        "posterior_mean": float(posterior_mean),
        "cred_int_low": float(cred_int_low),
        "cred_int_high": float(cred_int_high),
    }


if __name__ == "__main__":
    print(solve())