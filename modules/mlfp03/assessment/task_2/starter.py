# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP03 — Assessment Task 2: The Model Zoo

Complete the `solve()` function. Read problem.md for the full specification:
the six required algorithms, the deterministic data contract, and the exact
comparison-table schema. Train every model through the kailash-ml
`TrainingPipeline` (no raw `.fit()`). Your submission is auto-graded — the
grader independently re-trains one model to verify your table is real.

    python grader.py starter.py
"""
from __future__ import annotations

import asyncio
import warnings

import numpy as np
import polars as pl

from shared import MLFPDataLoader

warnings.filterwarnings("ignore")

N_ROWS = 10_000
SEED = 42
TARGET = "premium_response"

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

MODEL_ZOO: dict[str, tuple[str, str, dict]] = {
    "logistic_regression": (
        "sklearn.linear_model.LogisticRegression",
        "sklearn",
        {
            "max_iter": 2000,
            "random_state": SEED,
        },
    ),
    "naive_bayes": (
        "sklearn.naive_bayes.GaussianNB",
        "sklearn",
        {},
    ),
    "decision_tree": (
        "sklearn.tree.DecisionTreeClassifier",
        "sklearn",
        {
            "max_depth": 6,
            "random_state": SEED,
        },
    ),
    "random_forest": (
        "sklearn.ensemble.RandomForestClassifier",
        "sklearn",
        {
            "n_estimators": 150,
            "random_state": SEED,
            "n_jobs": -1,
        },
    ),
    "extra_trees": (
        "sklearn.ensemble.ExtraTreesClassifier",
        "sklearn",
        {
            "n_estimators": 150,
            "random_state": SEED,
            "n_jobs": -1,
        },
    ),
    "lightgbm": (
        "lightgbm.LGBMClassifier",
        "lightgbm",
        {
            "n_estimators": 200,
            "random_state": SEED,
            "verbose": -1,
        },
    ),
}


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


async def _run_zoo() -> list[dict]:
    """Train and evaluate every required model."""

    from kailash.db.connection import ConnectionManager
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
    # Build the deterministic modelling frame.
    # ============================================================
    frame = _model_frame()

    # ============================================================
    # TASK 2:
    # Define the FeatureSchema containing the eight predictors.
    # row_id is the entity ID and is not used as a predictor.
    # ============================================================
    schema = FeatureSchema(
        name="premium_upsell_features",
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

    # ============================================================
    # TASK 3:
    # Use a deterministic holdout evaluation for every model.
    # ============================================================
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

    rows: list[dict] = []

    try:
        await conn.initialize()

        registry = ModelRegistry(conn)

        pipeline = TrainingPipeline(
            feature_store=None,
            registry=registry,
        )

        # ========================================================
        # TASK 4:
        # Train all six models through TrainingPipeline.train().
        # Every model uses the same data and holdout split.
        # ========================================================
        for model_name, model_details in MODEL_ZOO.items():
            model_class, framework, hyperparameters = model_details

            model_spec = ModelSpec(
                model_class=model_class,
                framework=framework,
                hyperparameters=hyperparameters,
            )

            training_result = await pipeline.train(
                data=frame,
                schema=schema,
                model_spec=model_spec,
                eval_spec=eval_spec,
                experiment_name=model_name,
            )

            metrics = training_result.metrics

            rows.append(
                {
                    "model": model_name,
                    "accuracy": float(
                        metrics["accuracy"]
                    ),
                    "f1": float(
                        metrics["f1"]
                    ),
                    "auc": float(
                        metrics["auc"]
                    ),
                }
            )

    finally:
        await conn.close()

    return rows


def solve() -> pl.DataFrame:
    """Return the six-model comparison table sorted by AUC."""

    # ============================================================
    # TASK 5:
    # Run the asynchronous training function synchronously.
    # ============================================================
    rows = asyncio.run(
        _run_zoo()
    )

    # ============================================================
    # TASK 6:
    # Return the exact required schema sorted by AUC descending.
    # ============================================================
    result = (
        pl.DataFrame(
            rows,
            schema={
                "model": pl.String,
                "accuracy": pl.Float64,
                "f1": pl.Float64,
                "auc": pl.Float64,
            },
        )
        .select(
            [
                "model",
                "accuracy",
                "f1",
                "auc",
            ]
        )
        .sort(
            "auc",
            descending=True,
        )
    )

    return result


if __name__ == "__main__":
    print(solve())