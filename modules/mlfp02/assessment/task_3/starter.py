# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 3: Regression Modelling & Interpretation

Complete the `solve()` function. Read problem.md for the full specification.
Every regression is solved in closed form (OLS via least squares; logistic via
Newton-Raphson to the unique MLE) so your numbers must match the independently
re-derived reference. Standardise predictors (z-score) before fitting.

    python grader.py starter.py
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from shared import MLFPDataLoader

# --- Fixed problem constants (do not change) ---
OLS_FEATURES = [
    "income_imp",
    "age",
    "employment_years",
    "debt_to_income",
    "credit_age_years",
    "num_dependents",
    "edu_ord",
]
LOGIT_FEATURES = [
    "credit_utilization",
    "num_late_payments",
    "previous_defaults",
    "debt_to_income",
    "num_hard_inquiries",
]
EDU_MAP = {
    "primary": 1.0,
    "secondary": 2.0,
    "diploma": 3.0,
    "degree": 4.0,
    "postgraduate": 5.0,
}
TARGET = "loan_amount_sgd"


def solve() -> dict:
    """Return the regression / interpretation answer dict.

    See problem.md for the exact 13 keys and how each is defined.
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp02", "sg_credit_scoring.parquet")

    # TODO 1: Preprocess — median-impute income_sgd -> income_imp;
    #         ordinal-encode education via EDU_MAP -> edu_ord (Float64).
    #         No rows are dropped: n_obs == df.height.
    income_median = df.select(
        pl.col("income_sgd").median()
    ).item()

    df = df.with_columns(
        pl.col("income_sgd")
        .fill_null(income_median)
        .cast(pl.Float64)
        .alias("income_imp"),

        pl.col("education")
        .replace_strict(EDU_MAP)
        .cast(pl.Float64)
        .alias("edu_ord"),
    )

    n_obs = df.height

    # TODO 2: OLS — build the predictor matrix from OLS_FEATURES, z-score each
    #         column (population sd, ddof=0), prepend an intercept column of 1s,
    #         and solve beta via np.linalg.lstsq against the RAW target.
    ols_raw = (
        df
        .select(OLS_FEATURES)
        .to_numpy()
        .astype(float)
    )

    ols_means = np.mean(ols_raw, axis=0)
    ols_stds = np.std(ols_raw, axis=0, ddof=0)

    ols_std = (
        (ols_raw - ols_means)
        / ols_stds
    )

    X = np.column_stack(
        [
            np.ones(n_obs, dtype=float),
            ols_std,
        ]
    )

    y = (
        df
        .select(pl.col(TARGET).cast(pl.Float64))
        .to_series()
        .to_numpy()
        .astype(float)
    )

    beta, _, _, _ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    y_hat = X @ beta
    residuals = y - y_hat

    # TODO 3: Inference — rss, tss, r_squared, adj_r_squared,
    #         sigma2 = rss/(n-p), se = sqrt(diag(sigma2 * inv(X'X))),
    #         t = beta/se, p = 2*stats.t.sf(|t|, df=n-p),
    #         f_statistic = (r2/(p-1)) / ((1-r2)/(n-p)),
    #         f_p_value = stats.f.sf(f_statistic, p-1, n-p).
    #         Return coefficients / t_stats / p_values as dicts keyed by feature
    #         name plus "intercept".
    rss = np.sum(residuals ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)

    r_squared = 1.0 - rss / tss

    p = X.shape[1]
    residual_df = n_obs - p

    adj_r_squared = (
        1.0
        - (1.0 - r_squared)
        * (n_obs - 1)
        / residual_df
    )

    sigma2 = rss / residual_df

    xtx_inv = np.linalg.inv(X.T @ X)

    standard_errors = np.sqrt(
        np.diag(sigma2 * xtx_inv)
    )

    coefficient_t_stats = beta / standard_errors

    coefficient_p_values = (
        2.0
        * stats.t.sf(
            np.abs(coefficient_t_stats),
            df=residual_df,
        )
    )

    f_statistic = (
        (r_squared / (p - 1))
        / ((1.0 - r_squared) / residual_df)
    )

    f_p_value = stats.f.sf(
        f_statistic,
        p - 1,
        residual_df,
    )

    ols_names = ["intercept"] + OLS_FEATURES

    coefficients = {
        name: float(value)
        for name, value in zip(ols_names, beta)
    }

    t_stats = {
        name: float(value)
        for name, value in zip(
            ols_names,
            coefficient_t_stats,
        )
    }

    p_values = {
        name: float(value)
        for name, value in zip(
            ols_names,
            coefficient_p_values,
        )
    }

    # TODO 4: Partial F-test — add two terms built from the STANDARDISED base
    #         columns: income_std**2 and age_std*employment_std. Refit, then
    #         partial_f = ((rss - rss_full)/q) / (rss_full/(n - p_full)) with q=2;
    #         partial_f_p_value = stats.f.sf(partial_f, q, n - p_full);
    #         delta_r_squared = r2_full - r2.
    income_index = OLS_FEATURES.index("income_imp")
    age_index = OLS_FEATURES.index("age")
    employment_index = OLS_FEATURES.index("employment_years")

    income_squared = (
        ols_std[:, income_index] ** 2
    )

    age_employment_interaction = (
        ols_std[:, age_index]
        * ols_std[:, employment_index]
    )

    X_full = np.column_stack(
        [
            X,
            income_squared,
            age_employment_interaction,
        ]
    )

    beta_full, _, _, _ = np.linalg.lstsq(
        X_full,
        y,
        rcond=None,
    )

    y_hat_full = X_full @ beta_full
    residuals_full = y - y_hat_full

    rss_full = np.sum(residuals_full ** 2)

    r_squared_full = 1.0 - rss_full / tss

    q = 2
    p_full = X_full.shape[1]

    partial_f = (
        ((rss - rss_full) / q)
        / (rss_full / (n_obs - p_full))
    )

    partial_f_p_value = stats.f.sf(
        partial_f,
        q,
        n_obs - p_full,
    )

    delta_r_squared = (
        r_squared_full
        - r_squared
    )

    # TODO 5: Logistic — z-score LOGIT_FEATURES, prepend intercept, fit default
    #         via Newton-Raphson/IRLS to convergence; odds_ratios = exp(beta);
    #         strongest_logit_predictor = feature (excl. intercept) with max |beta|.
    logit_raw = (
        df
        .select(LOGIT_FEATURES)
        .to_numpy()
        .astype(float)
    )

    logit_means = np.mean(logit_raw, axis=0)
    logit_stds = np.std(logit_raw, axis=0, ddof=0)

    logit_std = (
        (logit_raw - logit_means)
        / logit_stds
    )

    X_logit = np.column_stack(
        [
            np.ones(n_obs, dtype=float),
            logit_std,
        ]
    )

    y_default = (
        df
        .select(pl.col("default").cast(pl.Float64))
        .to_series()
        .to_numpy()
        .astype(float)
    )

    logit_beta = np.zeros(
        X_logit.shape[1],
        dtype=float,
    )

    max_iterations = 100
    tolerance = 1e-10

    for _ in range(max_iterations):
        linear_predictor = X_logit @ logit_beta

        probabilities = stats.logistic.cdf(
            linear_predictor
        )

        weights = (
            probabilities
            * (1.0 - probabilities)
        )

        # Prevent numerical division or singularity if a probability becomes
        # extremely close to zero or one.
        weights = np.clip(
            weights,
            1e-12,
            None,
        )

        gradient = (
            X_logit.T
            @ (y_default - probabilities)
        )

        hessian_positive = (
            X_logit.T
            @ (X_logit * weights[:, None])
        )

        step = np.linalg.solve(
            hessian_positive,
            gradient,
        )

        new_beta = logit_beta + step

        if np.max(np.abs(new_beta - logit_beta)) < tolerance:
            logit_beta = new_beta
            break

        logit_beta = new_beta

    logit_names = ["intercept"] + LOGIT_FEATURES

    odds_ratio_values = np.exp(logit_beta)

    odds_ratios = {
        name: float(value)
        for name, value in zip(
            logit_names,
            odds_ratio_values,
        )
    }

    strongest_index = (
        np.argmax(np.abs(logit_beta[1:]))
    )

    strongest_logit_predictor = (
        LOGIT_FEATURES[strongest_index]
    )

    # TODO 6: Return the dict with all 13 keys (see problem.md).
    return {
        "n_obs": int(n_obs),
        "coefficients": coefficients,
        "t_stats": t_stats,
        "p_values": p_values,
        "r_squared": float(r_squared),
        "adj_r_squared": float(adj_r_squared),
        "f_statistic": float(f_statistic),
        "f_p_value": float(f_p_value),
        "partial_f": float(partial_f),
        "partial_f_p_value": float(partial_f_p_value),
        "delta_r_squared": float(delta_r_squared),
        "odds_ratios": odds_ratios,
        "strongest_logit_predictor": strongest_logit_predictor,
    }


if __name__ == "__main__":
    print(solve())