# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 2: HDB Feature Engineering

Complete the `solve()` function. Read problem.md for the full specification.
The raw strings are deliberately messy — read the data before you parse it.

    python grader.py starter.py
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


def solve() -> pl.DataFrame:
    """Engineer the 10-column feature table from raw HDB resale data.

    See problem.md for the exact columns, the storey_range OCR fix, the
    dual-format remaining_lease parser + null imputation rule, the room
    ordinal map, and the derived features.
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp01", "hdb_resale.parquet")

    # TODO 1: sale_year  <- first 4 chars of "month" ("YYYY-MM"), as Int.
    df = df.with_columns(
        pl.col("month").str.slice(0, 4).cast(pl.Int64).alias("sale_year")
    )

    # TODO 2: storey_midpoint  <- parse "LO TO HI"; some digits are OCR'd as the
    #         letter "O" (e.g. "4O TO 42"). Fix only the numeric tokens, then
    #         average the two bounds. (Careful: "TO" itself contains an O.)
    df = df.with_columns(
        (
            (
                pl.col("storey_range")
                .str.split_exact(" TO ", 1)
                .struct.field("field_0")
                .str.replace_all("O", "0")
                .cast(pl.Float64)
            )
            + (
                pl.col("storey_range")
                .str.split_exact(" TO ", 1)
                .struct.field("field_1")
                .str.replace_all("O", "0")
                .cast(pl.Float64)
            )
        )
        / 2
    ).alias("storey_midpoint")

    # TODO 3: flat_age_years  <- sale_year - lease_commence_date.
    df = df.with_columns(
        (
            pl.col("sale_year") - pl.col("lease_commence_date")
        ).cast(pl.Int64).alias("flat_age_years")
    )

    # TODO 4: price_per_sqm   <- resale_price / floor_area_sqm.
    df = df.with_columns(
        (
            pl.col("resale_price") / pl.col("floor_area_sqm")
        ).cast(pl.Float64).alias("price_per_sqm")
    )

    # TODO 5: flat_type_rooms <- ordinal map (2/3/4/5 ROOM -> 2..5,
    #         EXECUTIVE -> 6, MULTI-GENERATION -> 7).
    df = df.with_columns(
        pl.when(pl.col("flat_type") == "2 ROOM")
        .then(pl.lit(2))
        .when(pl.col("flat_type") == "3 ROOM")
        .then(pl.lit(3))
        .when(pl.col("flat_type") == "4 ROOM")
        .then(pl.lit(4))
        .when(pl.col("flat_type") == "5 ROOM")
        .then(pl.lit(5))
        .when(pl.col("flat_type") == "EXECUTIVE")
        .then(pl.lit(6))
        .when(pl.col("flat_type") == "MULTI-GENERATION")
        .then(pl.lit(7))
        .otherwise(None)
        .cast(pl.Int64)
        .alias("flat_type_rooms")
    )

    # TODO 6: remaining_lease_years <- parse BOTH "X years Y months" and bare
    #         "X"; impute nulls as 99 - flat_age_years (statutory 99y lease).
    df = df.with_columns(
        pl.when(pl.col("remaining_lease").is_null())
        .then(
            (pl.lit(99) - pl.col("flat_age_years")).cast(pl.Float64)
        )
        .otherwise(
            pl.col("remaining_lease")
            .str.extract(r"^(\d+)", 1)
            .cast(pl.Float64)
            + (
                pl.col("remaining_lease")
                .str.extract(r"(\d+)\s+months", 1)
                .cast(pl.Float64)
                .fill_null(0.0)
                / 12
            )
        )
        .alias("remaining_lease_years")
    )

    # TODO 7: select the 10 columns in order, sort by [sale_year, town].
    df = df.select(
        [
            "town",
            "flat_type",
            "flat_type_rooms",
            "sale_year",
            "storey_midpoint",
            "floor_area_sqm",
            "flat_age_years",
            "remaining_lease_years",
            "resale_price",
            "price_per_sqm",
        ]
    ).sort(["sale_year", "town"])

    return df  # <- replace with your 10-column engineered frame


if __name__ == "__main__":
    print(solve().head())