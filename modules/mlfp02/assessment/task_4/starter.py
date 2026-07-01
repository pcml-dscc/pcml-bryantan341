# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 4: Feature Engineering & Feature Store

Complete the `solve()` function. Read problem.md for the full specification.

    python grader.py starter.py
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader

DT_FMT = "%Y-%m-%d %H:%M:%S"

FEATURE_COLUMNS = [
    "admission_id",
    "feature_timestamp",
    "age",
    "gender",
    "bmi",
    "diagnosis",
    "icu_type",
    "mean_heart_rate",
    "mean_systolic_bp",
    "min_spo2",
    "max_temperature",
    "n_vitals",
    "n_labs",
    "n_abnormal_labs",
    "mean_creatinine",
    "n_distinct_drugs",
    "n_iv_meds",
    "total_dose_mg",
    "los_days",
]


def solve() -> pl.DataFrame:
    """Build the 19-column admission-level feature-store table."""

    loader = MLFPDataLoader()

    adm = loader.load(
        "mlfp02",
        "icu_admissions.parquet",
    )

    pat = loader.load(
        "mlfp02",
        "icu_patients.parquet",
    )

    vit = loader.load(
        "mlfp02",
        "icu_vitals.parquet",
    )

    labs = loader.load(
        "mlfp02",
        "icu_labs.parquet",
    )

    meds = loader.load(
        "mlfp02",
        "icu_medications.parquet",
    )

    # ============================================================
    # TASK 1:
    # Build the admissions base table.
    # Parse admit_time into feature_timestamp.
    # Left-join patient demographics.
    # ============================================================
    base = (
        adm
        .select(
            "admission_id",
            "patient_id",
            "diagnosis",
            "icu_type",
            "los_days",

            pl.col("admit_time")
            .str.strptime(
                pl.Datetime,
                format=DT_FMT,
                strict=True,
            )
            .alias("feature_timestamp"),
        )
        .join(
            pat.select(
                "patient_id",
                "age",
                "gender",
                "bmi",
            ),
            on="patient_id",
            how="left",
        )
    )

    # ============================================================
    # TASK 2:
    # Aggregate vitals by admission_id.
    # Calculate means, minimum, maximum and row count.
    # ============================================================
    vitals_agg = (
        vit
        .group_by("admission_id")
        .agg(
            pl.col("heart_rate")
            .mean()
            .alias("mean_heart_rate"),

            pl.col("systolic_bp")
            .mean()
            .alias("mean_systolic_bp"),

            pl.col("spo2")
            .min()
            .alias("min_spo2"),

            pl.col("temperature")
            .max()
            .alias("max_temperature"),

            pl.len()
            .cast(pl.Int64)
            .alias("n_vitals"),
        )
    )

    # ============================================================
    # TASK 3:
    # Parse lab values into Float64.
    # Junk strings become null because strict=False.
    # Lowercase the flag and aggregate labs by admission_id.
    # ============================================================
    labs_parsed = (
        labs
        .with_columns(
            pl.col("value")
            .cast(
                pl.Float64,
                strict=False,
            )
            .alias("parsed_value"),

            pl.col("flag")
            .str.to_lowercase()
            .alias("flag_lower"),
        )
    )

    labs_agg = (
        labs_parsed
        .group_by("admission_id")
        .agg(
            pl.len()
            .cast(pl.Int64)
            .alias("n_labs"),

            (
                pl.col("flag_lower")
                == "abnormal"
            )
            .sum()
            .cast(pl.Int64)
            .alias("n_abnormal_labs"),

            pl.col("parsed_value")
            .filter(
                pl.col("test_name")
                == "Creatinine"
            )
            .mean()
            .alias("mean_creatinine"),
        )
    )

    # ============================================================
    # TASK 4:
    # Extract the numeric part of medication dose.
    # Aggregate medication information by admission_id.
    # ============================================================
    meds_parsed = (
        meds
        .with_columns(
            pl.col("dose")
            .str.extract(
                r"([0-9]+\.?[0-9]*)",
                group_index=1,
            )
            .cast(
                pl.Float64,
                strict=False,
            )
            .alias("dose_mg"),
        )
    )

    meds_agg = (
        meds_parsed
        .group_by("admission_id")
        .agg(
            pl.col("drug_name")
            .n_unique()
            .cast(pl.Int64)
            .alias("n_distinct_drugs"),

            (
                pl.col("route") == "IV"
            )
            .sum()
            .cast(pl.Int64)
            .alias("n_iv_meds"),

            pl.col("dose_mg")
            .sum()
            .alias("total_dose_mg"),
        )
    )

    # ============================================================
    # TASK 5:
    # Left-join all aggregated feature tables onto the base table.
    # ============================================================
    result = (
        base
        .join(
            vitals_agg,
            on="admission_id",
            how="left",
        )
        .join(
            labs_agg,
            on="admission_id",
            how="left",
        )
        .join(
            meds_agg,
            on="admission_id",
            how="left",
        )
    )

    # ============================================================
    # TASK 6:
    # Apply the required imputation rules.
    # Calculate all medians before filling any null values.
    # ============================================================
    count_columns = [
        "n_vitals",
        "n_labs",
        "n_abnormal_labs",
        "n_distinct_drugs",
        "n_iv_meds",
    ]

    median_columns = [
        "age",
        "bmi",
        "mean_heart_rate",
        "mean_systolic_bp",
        "min_spo2",
        "max_temperature",
        "mean_creatinine",
    ]

    median_values = (
        result
        .select(
            [
                pl.col(column)
                .cast(pl.Float64)
                .median()
                .alias(column)
                for column in median_columns
            ]
        )
        .row(
            0,
            named=True,
        )
    )

    result = result.with_columns(
        # Fill missing gender.
        pl.col("gender")
        .fill_null("Unknown")
        .alias("gender"),

        # Fill missing total dose with 0.0.
        pl.col("total_dose_mg")
        .cast(pl.Float64)
        .fill_null(0.0)
        .alias("total_dose_mg"),

        # Fill missing count columns with 0.
        *[
            pl.col(column)
            .fill_null(0)
            .cast(pl.Int64)
            .alias(column)
            for column in count_columns
        ],

        # Fill numeric measurement columns using their medians.
        *[
            pl.col(column)
            .cast(pl.Float64)
            .fill_null(
                float(median_values[column])
            )
            .alias(column)
            for column in median_columns
        ],
    )

    # ============================================================
    # TASK 7:
    # Select the exact 19 required columns in the correct order.
    # Sort the result by admission_id.
    # ============================================================
    result = (
        result
        .select(FEATURE_COLUMNS)
        .sort("admission_id")
    )

    return result


if __name__ == "__main__":
    print(solve().head())