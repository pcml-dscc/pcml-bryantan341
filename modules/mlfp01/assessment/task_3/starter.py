# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 3: Window Functions & Price Trends

Complete the `solve()` function. Read problem.md for the full specification.
This task is about correct window partitioning and ordering.

    python grader.py starter.py
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


def solve() -> pl.DataFrame:
    """Per-town, per-year HDB price-trend table (7 columns).

    See problem.md for the exact columns and the four window computations
    (YoY % within town, 3-year rolling average within town, rank within year).
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp01", "hdb_resale.parquet")

    # TODO 1: derive sale_year from "month" ("YYYY-MM").
    df = df.with_columns(
        pl.col("month")
        .str.slice(0, 4)
        .cast(pl.Int64)
        .alias("sale_year")
    )

    # TODO 2: aggregate to one row per (town, sale_year): median_price =
    #         median(resale_price), n_sales = row count.
    df = (
        df.group_by(["town", "sale_year"])
        .agg(
            pl.col("resale_price").median().alias("median_price"),
            pl.len().alias("n_sales"),
        )
        .sort(["town", "sale_year"])
    )

    # TODO 3: yoy_pct  <- 100 * (median - prev_year_median) / prev_year_median,
    #         computed WITHIN each town ordered by year (null for first year).
    df = df.with_columns(
        pl.col("median_price")
        .shift(1)
        .over("town")
        .alias("prev_year_median")
    )

    df = df.with_columns(
        (
            100
            * (
                pl.col("median_price")
                - pl.col("prev_year_median")
            )
            / pl.col("prev_year_median")
        ).alias("yoy_pct")
    )

    # TODO 4: rolling_3yr_avg <- 3-year trailing mean of median_price WITHIN
    #         town (min_periods=1).
    df = df.with_columns(
        pl.col("median_price")
        .rolling_mean(window_size=3, min_samples=1)
        .over("town")
        .alias("rolling_3yr_avg")
    )

    # TODO 5: price_rank_in_year <- rank of median_price WITHIN each year,
    #         descending so 1 = most expensive town (method="min").
    df = df.with_columns(
        pl.col("median_price")
        .rank(method="min", descending=True)
        .over("sale_year")
        .cast(pl.Int64)
        .alias("price_rank_in_year")
    )

    # TODO 6: select the 7 colummns in order, sort by [town, sale_year].
    df = df.select(
        [
            "town",
            "sale_year",
            "n_sales",
            "median_price",
            "yoy_pct",
            "rolling_3yr_avg",
            "price_rank_in_year",
        ]
    ).sort(["town", "sale_year"])

    return df  # <- replace with your 7-column trend table


if __name__ == "__main__":
    print(solve().head())