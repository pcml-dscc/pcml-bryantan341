# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP04 — Assessment Task 2: Dimensionality Reduction & Anomaly Detection

Complete the `solve()` function. Read problem.md for the full specification.
Framework-first: use DimReductionEngine (PCA) and AnomalyDetectionEngine.

    python grader.py starter.py     # grade your attempt
"""
from __future__ import annotations

import numpy as np
import polars as pl

from kailash_ml.engines.anomaly_detection import AnomalyDetectionEngine
from kailash_ml.engines.dim_reduction import DimReductionEngine

SEED = 20260402
N_NORMAL = 975
N_ANOM = 25
D = 24
K_LATENT = 3
CONTAMINATION = 0.025


def make_sensor_matrix() -> pl.DataFrame:
    """Deterministic 24-channel telemetry on a 3-factor manifold + 25 outliers.

    Do NOT change the seed or sizes — the grader regenerates this exact table
    (and the hidden anomaly flags it never gave you).
    """
    rng = np.random.default_rng(SEED)
    Z = rng.normal(0, 1, (N_NORMAL, K_LATENT))
    W = rng.normal(0, 1, (K_LATENT, D)) * 3.5
    X_normal = Z @ W + rng.normal(0, 0.5, (N_NORMAL, D))
    X_anom = rng.normal(12.0, 4.0, (N_ANOM, D)) * rng.choice([-1, 1], (N_ANOM, D))
    X = np.vstack([X_normal, X_anom])
    perm = rng.permutation(X.shape[0])
    X = X[perm]
    cols = [f"f{i:02d}" for i in range(D)]
    return pl.DataFrame({c: X[:, j] for j, c in enumerate(cols)})


def solve() -> dict:
    """Compress with PCA and flag off-manifold rows — kailash-ml engines."""

    # Task 1: Generate the deterministic sensor dataset
    df = make_sensor_matrix()

    # Task 2: Find the minimum number of PCA components that explain at least 90% variance
    dim_engine = DimReductionEngine()

    full_result = dim_engine.reduce(
        df,
        algorithm="pca",
        n_components=df.width,
    )

    explained = np.array(full_result.explained_variance_ratio)
    cumulative = np.cumsum(explained)
    n_components_90 = int(np.argmax(cumulative >= 0.90) + 1)

    # Task 3: Compress the dataset using the selected number of components
    reduced_result = dim_engine.reduce(
        df,
        algorithm="pca",
        n_components=n_components_90,
    )

    reconstruction_error = float(reduced_result.reconstruction_error)

    # Task 4: Detect anomalies using Isolation Forest
    anomaly_engine = AnomalyDetectionEngine()

    anomaly_result = anomaly_engine.detect(
        df,
        algorithm="isolation_forest",
        contamination=CONTAMINATION,
    )

    anomaly_scores = list(anomaly_result.scores)
    anomaly_labels = [1 if label == -1 else 0 for label in anomaly_result.labels]
    n_anomalies = int(anomaly_result.n_anomalies)

    # Task 5: Return the required output
    return {
        "n_components_90": n_components_90,
        "reconstruction_error": reconstruction_error,
        "anomaly_scores": anomaly_scores,
        "anomaly_labels": anomaly_labels,
        "n_anomalies": n_anomalies,
    }


if __name__ == "__main__":
    print(solve())