# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP03 — Assessment Task 4: Production Pipeline — Registry, Drift, Deploy

Complete the `solve()` function. Read problem.md for the full specification.
Train a LightGBM model through the kailash-ml `TrainingPipeline`, register and
promote it staging -> production in the `ModelRegistry`, then arm a
`DriftMonitor` and check a clean slice (no alarm) and a shifted slice (drift).
Your submission is auto-graded against an independent re-derivation.

    python grader.py starter.py

IMPORTANT — TWO DATABASES: give the ModelRegistry and the DriftMonitor SEPARATE
SQLite files. A model registry and a monitoring store are distinct systems with
independent lifecycles, and using fresh, separate files avoids reusing a stale
database whose schema predates your installed kailash-ml version.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from shared import MLFPDataLoader

warnings.filterwarnings("ignore")

N_ROWS = 10_000
SEED = 42
TARGET = "premium_response"
REFERENCE_ROWS = 7_500
PSI_THRESHOLD = 0.2
KS_THRESHOLD = 0.05

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


def _shift_slice(
    clean: pl.DataFrame,
) -> pl.DataFrame:
    """Apply the required economic-downturn shift."""

    return clean.with_columns(
        [
            (
                pl.col("avg_order_value")
                * 0.6
            ).alias("avg_order_value"),

            (
                pl.col("total_revenue")
                * 0.6
            ).alias("total_revenue"),

            (
                pl.col("days_since_last_order")
                * 1.5
                + 60
            ).alias("days_since_last_order"),

            (
                pl.col("satisfaction_score")
                - 1
            ).alias("satisfaction_score"),
        ]
    )


async def _run() -> dict:
    """Train, register, promote and monitor the model."""

    from kailash.db.connection import ConnectionManager
    from kailash_ml.engines.drift_monitor import DriftMonitor
    from kailash_ml.engines.model_registry import (
        LocalFileArtifactStore,
        ModelRegistry,
    )
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
    # Build the model frame and the three monitoring datasets.
    # ============================================================
    frame = _model_frame()

    reference = (
        frame
        .select(BASE_FEATURES)
        .head(REFERENCE_ROWS)
    )

    clean = (
        frame
        .select(BASE_FEATURES)
        .tail(
            frame.height
            - REFERENCE_ROWS
        )
    )

    shifted = _shift_slice(
        clean
    )

    # ============================================================
    # TASK 2:
    # Define the feature schema.
    # ============================================================
    schema = FeatureSchema(
        name="premium_upsell_production_features",
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
    # Create fresh and separate temporary paths for:
    # 1. Model registry database
    # 2. Drift monitor database
    # 3. Model artifact directory
    # ============================================================
    unique_id = (
        f"{os.getpid()}_"
        f"{uuid.uuid4().hex}"
    )

    temp_root = Path(
        tempfile.gettempdir()
    )

    registry_db_path = (
        temp_root
        / f"mlfp03_registry_{unique_id}.db"
    )

    drift_db_path = (
        temp_root
        / f"mlfp03_drift_{unique_id}.db"
    )

    artifact_path = (
        temp_root
        / f"mlfp03_artifacts_{unique_id}"
    )

    registry_conn = ConnectionManager(
        f"sqlite:///{registry_db_path.as_posix()}"
    )

    drift_conn = ConnectionManager(
        f"sqlite:///{drift_db_path.as_posix()}"
    )

    try:
        await registry_conn.initialize()
        await drift_conn.initialize()

        # ========================================================
        # TASK 4:
        # Create the registry and TrainingPipeline using only the
        # registry database connection.
        # ========================================================
        artifact_store = LocalFileArtifactStore(
            artifact_path
        )

        registry = ModelRegistry(
            registry_conn,
            artifact_store=artifact_store,
        )

        pipeline = TrainingPipeline(
            feature_store=None,
            registry=registry,
        )

        model_name = "premium_upsell_lightgbm"

        model_spec = ModelSpec(
            model_class="lightgbm.LGBMClassifier",
            framework="lightgbm",
            hyperparameters={
                "n_estimators": 200,
                "random_state": SEED,
                "verbose": -1,
            },
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

        # ========================================================
        # TASK 5:
        # Train and automatically register the model at staging.
        # ========================================================
        training_result = await pipeline.train(
            data=frame,
            schema=schema,
            model_spec=model_spec,
            eval_spec=eval_spec,
            experiment_name=model_name,
        )

        registered_model = (
            training_result.model_version
        )

        if registered_model is None:
            raise RuntimeError(
                "The trained model was not registered."
            )

        registered_version = int(
            registered_model.version
        )

        reference_auc = float(
            training_result.metrics["auc"]
        )

        # ========================================================
        # TASK 6:
        # Promote staging -> production with an audit reason.
        # Retrieve the production model to confirm promotion.
        # ========================================================
        await registry.promote_model(
            name=model_name,
            version=registered_version,
            target_stage="production",
            reason=(
                "Passed holdout evaluation and approved "
                "for premium-upsell production deployment."
            ),
        )

        production_model = await registry.get_model(
            model_name,
            stage="production",
        )

        production_stage = (
            production_model.stage
        )

        # ========================================================
        # TASK 7:
        # Create DriftMonitor using the separate drift database.
        # Set the reference distribution.
        # ========================================================
        monitor = DriftMonitor(
            drift_conn,
            tenant_id="_single",
            psi_threshold=PSI_THRESHOLD,
            ks_threshold=KS_THRESHOLD,
        )

        await monitor.set_reference_data(
            model_name,
            reference,
            BASE_FEATURES,
        )

        # ========================================================
        # TASK 8:
        # Check the clean batch and shifted batch for drift.
        # ========================================================
        clean_report = await monitor.check_drift(
            model_name,
            clean,
        )

        shift_report = await monitor.check_drift(
            model_name,
            shifted,
        )

        clean_drift_detected = bool(
            clean_report.overall_drift_detected
        )

        shift_drift_detected = bool(
            shift_report.overall_drift_detected
        )

        n_drifted_features_clean = len(
            clean_report.drifted_features
        )

        n_drifted_features_shift = len(
            shift_report.drifted_features
        )

        shift_severity = (
            shift_report.overall_severity
        )

        # ========================================================
        # TASK 9:
        # Return the exact required production-lifecycle result.
        # ========================================================
        return {
            "registered_version": registered_version,
            "production_stage": production_stage,
            "reference_auc": reference_auc,
            "clean_drift_detected": clean_drift_detected,
            "shift_drift_detected": shift_drift_detected,
            "n_drifted_features_clean": int(
                n_drifted_features_clean
            ),
            "n_drifted_features_shift": int(
                n_drifted_features_shift
            ),
            "shift_severity": shift_severity,
        }

    finally:
        # ========================================================
        # TASK 10:
        # Close both independent database connections.
        # Remove all temporary databases and model artifacts.
        # ========================================================
        await registry_conn.close()
        await drift_conn.close()

        registry_sidecars = [
            registry_db_path,
            Path(str(registry_db_path) + "-wal"),
            Path(str(registry_db_path) + "-shm"),
        ]

        drift_sidecars = [
            drift_db_path,
            Path(str(drift_db_path) + "-wal"),
            Path(str(drift_db_path) + "-shm"),
        ]

        for path in (
            registry_sidecars
            + drift_sidecars
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

        if artifact_path.exists():
            import shutil

            shutil.rmtree(
                artifact_path,
                ignore_errors=True,
            )


def solve() -> dict:
    """Run the asynchronous production pipeline."""

    return asyncio.run(
        _run()
    )


if __name__ == "__main__":
    print(solve())