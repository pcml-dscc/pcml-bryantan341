# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 4: Feature Engineering & Feature Store

Complete the `solve()` function. Read problem.md for the full specification.
You join five raw ICU tables into one admission-level feature table. The event
tables are messy: lab values contain junk strings, doses are like "34.8MG", and
many admissions have no vitals/labs at all. Your output is auto-graded
column-by-column against an independent re-derivation.

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
    """Build the 19-column admission-level feature-store table.

    See problem.md for the exact columns, aggregations, parsing rules, and the
    imputation policy. Return the table sorted ascending by admission_id.
    """
    loader = MLFPDataLoader()
    adm = loader.load("mlfp02", "icu_admissions.parquet")
    pat = loader.load("mlfp02", "icu_patients.parquet")
    vit = loader.load("mlfp02", "icu_vitals.parquet")
    labs = loader.load("mlfp02", "icu_labs.parquet")
    meds = loader.load("mlfp02", "icu_medications.parquet")

    # TODO 1: Base = admissions with admission_id, patient_id, diagnosis,
    #         icu_type, los_days, and feature_timestamp = admit_time parsed to
    #         Datetime (format DT_FMT). Left-join patient age, gender, bmi on
    #         patient_id.
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

    # TODO 2: Vitals -> group_by admission_id: mean_heart_rate, mean_systolic_bp,
    #         min_spo2, max_temperature, n_vitals = count of rows.
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

    # TODO 3: Labs -> parse value to Float64 (strict=False so junk like
    #         "HAEMOLYSED"/"<0.1" becomes null); lowercase flag. group_by
    #         admission_id: n_labs = row count, n_abnormal_labs = count where
    #         flag == "abnormal", mean_creatinine = mean parsed value where
    #         test_name == "Creatinine".
    labs_parsed = (
        labs
        .with_columns(
            pl.col("value")
            .cast(pl.Float64, strict=False)
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

            (pl.col("flag_lower") == "abnormal")
            .sum()
            .cast(pl.Int64)
            .alias("n_abnormal_labs"),

            pl.col("parsed_value")
            .filter(pl.col("test_name") == "Creatinine")
            .mean()
            .alias("mean_creatinine"),
        )
    )

    # TODO 4: Medications -> parse leading numeric of dose via regex
    #         r"([0-9]+\.?[0-9]*)" -> Float64 mg. group_by admission_id:
    #         n_distinct_drugs = n_unique(drug_name), n_iv_meds = count where
    #         route == "IV", total_dose_mg = sum of parsed dose.
    meds_parsed = (
        meds
        .with_columns(
            pl.col("dose")
            .str.extract(
                r"([0-9]+\.?[0-9]*)",
                group_index=1,
            )
            .cast(pl.Float64, strict=False)
            .alias("dose_mg")
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

            (pl.col("route") == "IV")
            .sum()
            .cast(pl.Int64)
            .alias("n_iv_meds"),

            pl.col("dose_mg")
            .sum()
            .alias("total_dose_mg"),
        )
    )

    # TODO 5: Left-join all three aggregate blocks onto the base.
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

    # TODO 6: Imputation policy:
    #           - gender null -> "Unknown"
    #           - total_dose_mg null -> 0.0
    #           - n_vitals/n_labs/n_abnormal_labs/n_distinct_drugs/n_iv_meds
    #             null -> 0 (cast to Int64)
    #           - age, bmi, mean_heart_rate, mean_systolic_bp, min_spo2,
    #             max_temperature, mean_creatinine null -> that column's MEDIAN
    #             (computed before filling; cast to Float64)
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

    # Compute all medians before filling the missing values.
    medians = (
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
        .row(0, named=True)
    )

    result = result.with_columns(
        pl.col("gender")
        .fill_null("Unknown")
        .alias("gender"),

        pl.col("total_dose_mg")
        .cast(pl.Float64)
        .fill_null(0.0)
        .alias("total_dose_mg"),

        *[
            pl.col(column)
            .fill_null(0)
            .cast(pl.Int64)
            .alias(column)
            for column in count_columns
        ],

        *[
            pl.col(column)
            .cast(pl.Float64)
            .fill_null(float(medians[column]))
            .alias(column)
            for column in median_columns
        ],
    )

    # TODO 7: select FEATURE_COLUMNS in order, sort by admission_id.
    result = (
        result
        .select(FEATURE_COLUMNS)
        .sort("admission_id")
    )

    return result


if __name__ == "__main__":
    print(solve().head())