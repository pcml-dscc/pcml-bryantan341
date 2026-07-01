# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP03 — Assessment Task 3: Evaluation, Class Imbalance & Interpretability

Complete the `solve()` function. Read problem.md for the full specification.
Train a baseline and a class-balanced RandomForest through the kailash-ml
`TrainingPipeline`, evaluate per-class behaviour with `km.diagnose`, and explain
the balanced model with `ModelExplainer` (SHAP). Your submission is auto-graded
against an independent re-derivation.

    python grader.py starter.py
"""
from __future__ import annotations

import asyncio
import pickle
import warnings

import numpy as np
import polars as pl

from shared import MLFPDataLoader

warnings.filterwarnings("ignore")

N_ROWS = 10_000
SEED = 42
TARGET = "premium_response"
TOP_K = 6
SHAP_BACKGROUND = 64

BASE_FEATURES = [
    "satisfaction_score",
    "avg_order_value",
    "num_returns",
    "order_count",
    "loyalty_int",
    "total_revenue",
    "days_since_last_order",
    "customer_tenure_days",
]


def _model_frame() -> pl.DataFrame:
    """Load the data and derive premium_response."""

    df = MLFPDataLoader().load(
        "mlfp03",
        "ecommerce_customers.parquet",
    )

    df = (
        df
        .sort("customer_id")
        .head(N_ROWS)
    )

    rng = np.random.default_rng(SEED)

    def z(col: str) -> np.ndarray:
        a = df[col].to_numpy().astype(float)
        return (a - a.mean()) / (a.std() + 1e-9)

    loyal = (
        df["loyalty_member"]
        .cast(pl.Int64)
        .to_numpy()
        .astype(float)
    )

    sat_high = (
        (df["satisfaction_score"] >= 4)
        .cast(pl.Int64)
        .to_numpy()
        .astype(float)
    )

    logit = (
        1.0 * z("satisfaction_score")
        + 0.9 * loyal
        + 0.8 * z("avg_order_value")
        - 0.7 * z("num_returns")
        + 0.5 * z("order_count")
        + 1.4 * (loyal * sat_high)
        + rng.normal(0.0, 1.3, size=df.height)
    )

    df = df.with_columns(
        [
            pl.col("loyalty_member")
            .cast(pl.Int64)
            .alias("loyalty_int"),

            pl.Series(
                TARGET,
                (logit > 2.0).astype(np.int64),
            ),

            pl.int_range(
                0,
                df.height,
                dtype=pl.Int64,
            ).alias("row_id"),
        ]
    )

    return df.select(
        BASE_FEATURES
        + ["row_id", TARGET]
    )


def _holdout_test(
    frame: pl.DataFrame,
) -> pl.DataFrame:
    """Reproduce TrainingPipeline's test split."""

    n = frame.height

    idx = np.arange(n)

    np.random.RandomState(
        SEED
    ).shuffle(idx)

    split_idx = int(
        n * 0.75
    )

    return frame[
        idx[split_idx:].tolist()
    ]


def _minority_recall(
    report,
) -> float:
    """Read positive-class recall safely."""

    if "1.0" in report.per_class:
        return float(
            report.per_class["1.0"]["recall"]
        )

    return float(
        report.per_class["1"]["recall"]
    )


async def _run() -> dict:
    """Train, diagnose and explain the two RandomForest models."""

    from kailash.db.connection import ConnectionManager
    from kailash_ml import diagnose
    from kailash_ml.engines.model_explainer import ModelExplainer
    from kailash_ml.engines.model_registry import ModelRegistry
    from kailash_ml.engines.training_pipeline import (
        EvalSpec,
        ModelSpec,
        TrainingPipeline,
    )
    from kailash_ml.types import (
        FeatureField,
        FeatureSchema,
    )

    # ============================================================
    # TASK 1:
    # Build the deterministic model frame and feature schema.
    # ============================================================
    frame = _model_frame()

    schema = FeatureSchema(
        name="premium_upsell_imbalance_features",
        features=[
            FeatureField(
                name=feature,
                dtype="float64",
                nullable=False,
            )
            for feature in BASE_FEATURES
        ],
        entity_id_column="row_id",
    )

    eval_spec = EvalSpec(
        metrics=[
            "accuracy",
            "f1",
            "auc",
        ],
        split_strategy="holdout",
        test_size=0.25,
    )

    conn = ConnectionManager(
        "sqlite:///:memory:"
    )

    try:
        await conn.initialize()

        registry = ModelRegistry(conn)

        pipeline = TrainingPipeline(
            feature_store=None,
            registry=registry,
        )

        # ========================================================
        # TASK 2:
        # Train the baseline RandomForest.
        # ========================================================
        baseline_spec = ModelSpec(
            model_class=(
                "sklearn.ensemble."
                "RandomForestClassifier"
            ),
            framework="sklearn",
            hyperparameters={
                "n_estimators": 150,
                "random_state": SEED,
                "n_jobs": -1,
            },
        )

        baseline_result = await pipeline.train(
            data=frame,
            schema=schema,
            model_spec=baseline_spec,
            eval_spec=eval_spec,
            experiment_name="premium_rf_baseline",
        )

        # ========================================================
        # TASK 3:
        # Train the balanced RandomForest.
        # ========================================================
        balanced_spec = ModelSpec(
            model_class=(
                "sklearn.ensemble."
                "RandomForestClassifier"
            ),
            framework="sklearn",
            hyperparameters={
                "n_estimators": 150,
                "random_state": SEED,
                "n_jobs": -1,
                "class_weight": "balanced",
            },
        )

        balanced_result = await pipeline.train(
            data=frame,
            schema=schema,
            model_spec=balanced_spec,
            eval_spec=eval_spec,
            experiment_name="premium_rf_balanced",
        )

        # ========================================================
        # TASK 4:
        # Load both fitted models from the ModelRegistry.
        # ========================================================
        baseline_version = (
            baseline_result.model_version
        )

        balanced_version = (
            balanced_result.model_version
        )

        if baseline_version is None:
            raise RuntimeError(
                "Baseline model was not registered."
            )

        if balanced_version is None:
            raise RuntimeError(
                "Balanced model was not registered."
            )

        baseline_artifact = await registry.load_artifact(
            baseline_version.name,
            baseline_version.version,
        )

        balanced_artifact = await registry.load_artifact(
            balanced_version.name,
            balanced_version.version,
        )

        baseline_model = pickle.loads(
            baseline_artifact
        )

        balanced_model = pickle.loads(
            balanced_artifact
        )

        # ========================================================
        # TASK 5:
        # Reproduce the exact test split and diagnose both models.
        # ========================================================
        test = _holdout_test(
            frame
        )

        X_test = test.select(
            BASE_FEATURES
        )

        # Cast to Float64 so the diagnostic class key is "1.0".
        y_test = test[TARGET].cast(
            pl.Float64
        )

        baseline_report = diagnose(
            baseline_model,
            kind="classical_classifier",
            data=(
                X_test,
                y_test,
            ),
            show=False,
        )

        balanced_report = diagnose(
            balanced_model,
            kind="classical_classifier",
            data=(
                X_test,
                y_test,
            ),
            show=False,
        )

        baseline_minority_recall = _minority_recall(
            baseline_report
        )

        balanced_minority_recall = _minority_recall(
            balanced_report
        )

        baseline_recall_macro = float(
            baseline_report.metrics[
                "recall_macro"
            ]
        )

        balanced_recall_macro = float(
            balanced_report.metrics[
                "recall_macro"
            ]
        )

        baseline_accuracy = float(
            baseline_report.metrics[
                "accuracy"
            ]
        )

        balanced_accuracy = float(
            balanced_report.metrics[
                "accuracy"
            ]
        )

        # ========================================================
        # TASK 6:
        # Explain the balanced model using SHAP global importance.
        # ========================================================
        background = (
            frame
            .select(BASE_FEATURES)
            .head(SHAP_BACKGROUND)
        )

        explainer = ModelExplainer(
            model=balanced_model,
            X=background,
            feature_names=BASE_FEATURES,
        )

        explanation = explainer.explain_global(
            max_display=TOP_K
        )

        feature_importance = explanation[
            "feature_importance"
        ]

        top_features = list(
            feature_importance.keys()
        )[:TOP_K]

        # ========================================================
        # TASK 7:
        # Return all required metrics and top features.
        # ========================================================
        return {
            "baseline_minority_recall": float(
                baseline_minority_recall
            ),
            "balanced_minority_recall": float(
                balanced_minority_recall
            ),
            "baseline_recall_macro": float(
                baseline_recall_macro
            ),
            "balanced_recall_macro": float(
                balanced_recall_macro
            ),
            "baseline_accuracy": float(
                baseline_accuracy
            ),
            "balanced_accuracy": float(
                balanced_accuracy
            ),
            "roc_auc": float(
                balanced_result.metrics["auc"]
            ),
            "top_features": top_features,
            "n_features": len(BASE_FEATURES),
        }

    finally:
        await conn.close()


def solve() -> dict:
    """Run the asynchronous evaluation pipeline."""

    return asyncio.run(
        _run()
    )


if __name__ == "__main__":
    print(solve())