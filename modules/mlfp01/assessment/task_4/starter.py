# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 4: Profile, Clean & Integrate with DataExplorer

Complete the `solve()` function. Read problem.md for the full specification.
This task uses the kailash-ml DataExplorer engine to PROVE your cleaning
improved data quality.

    python grader.py starter.py
"""
from __future__ import annotations

import asyncio

import polars as pl

from kailash_ml import DataExplorer
from shared import MLFPDataLoader


def solve() -> dict:
    """Return {"cleaned": pl.DataFrame, "raw_alert_count": int, "clean_alert_count": int}.

    See problem.md for the exact 8-column cleaned schema, the THREE period
    formats you must parse, the comma-stripped integer cast, the median
    imputation rule, and the DataExplorer alert-count requirement.
    """
    raw = MLFPDataLoader().load("mlfp01", "economic_indicators.csv")

    # TODO 1: keep only period_type == "quarterly".
    quarterly = raw.filter(
        pl.col("period_type") == "quarterly"
    )

    # TODO 2: parse period into period_year (Int) + period_quarter (Int, 1-4).
    #         It appears in THREE formats: "Q1 2000", "2001-Q1", and "2001-2".
    cleaned = quarterly.with_columns(
        pl.coalesce(
            pl.col("period").str.extract(r"^Q[1-4]\s+(\d{4})$", 1),
            pl.col("period").str.extract(r"^(\d{4})-", 1),
        )
        .cast(pl.Int64)
        .alias("period_year"),

        pl.coalesce(
            pl.col("period").str.extract(r"^Q([1-4])\s+\d{4}$", 1),
            pl.col("period").str.extract(r"^\d{4}-Q([1-4])$", 1),
            pl.col("period").str.extract(r"^\d{4}-([1-4])$", 1),
        )
        .cast(pl.Int64)
        .alias("period_quarter"),
    )

    # TODO 3: tourist_arrivals -> Int64 (strip thousands separators like "5,246,242").
    cleaned = cleaned.with_columns(
        pl.col("tourist_arrivals")
        .str.replace_all(",", "")
        .cast(pl.Int64)
        .alias("tourist_arrivals")
    )

    # TODO 4: impute inflation_rate and trade_balance_sgd_bn nulls with the
    #         quarterly median of each column.
    cleaned = cleaned.with_columns(
        pl.col("inflation_rate")
        .fill_null(pl.col("inflation_rate").median())
        .alias("inflation_rate"),

        pl.col("trade_balance_sgd_bn")
        .fill_null(pl.col("trade_balance_sgd_bn").median())
        .alias("trade_balance_sgd_bn"),
    )

    # TODO 5: select the 8 columns (see problem.md) sorted by [period_year, period_quarter].
    cleaned = cleaned.select(
        [
            "period_year",
            "period_quarter",
            "gdp_growth_pct",
            "unemployment_rate",
            "inflation_rate",
            "trade_balance_sgd_bn",
            "property_price_index",
            "tourist_arrivals",
        ]
    ).sort(
        ["period_year", "period_quarter"]
    )

    # TODO 6: profile the RAW quarterly slice AND your cleaned frame with
    #         DataExplorer; count alerts on each (await explorer.profile(df);
    #         use len(profile.alerts)). Cleaning must REDUCE the alert count.
    async def profile_frames() -> tuple[int, int]:
        explorer = DataExplorer()

        raw_profile = await explorer.profile(quarterly)
        clean_profile = await explorer.profile(cleaned)

        return len(raw_profile.alerts), len(clean_profile.alerts)

    raw_alert_count, clean_alert_count = asyncio.run(profile_frames())

    return {
        "cleaned": cleaned,
        "raw_alert_count": raw_alert_count,
        "clean_alert_count": clean_alert_count,
    }


if __name__ == "__main__":
    print(solve())