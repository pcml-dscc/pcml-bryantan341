# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP04 — Assessment Task 3: NLP Topic Discovery with NMF

Complete the `solve()` function. Read problem.md for the full specification.
Framework-first: factor the document-term matrix with DimReductionEngine (NMF).
The TF-IDF helper `build_tfidf` is provided — use it as given.

    python grader.py starter.py     # grade your attempt
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import polars as pl

from kailash_ml.engines.dim_reduction import DimReductionEngine
from shared import MLFPDataLoader

CATEGORIES = ["finance", "food", "geography", "transport"]
N_TOPICS = 4

STOPWORDS = set(
    "the a an and or but of to in on at for with as is are was were be been being "
    "this that these those it its they them their from by we you your our he she "
    "his her not have has had do does did will would can could should about into "
    "over under more most some any all than then so such which who what when where "
    "how also use used using many much each both other one two new high low first "
    "second within across per via etc".split()
)
STOPWORDS |= {
    "singapore",
    "singapores",
    "singaporean",
    "singaporeans",
    "country",
    "city",
    "include",
    "including",
    "main",
    "known",
    "provides",
    "provide",
    "offers",
    "offer",
}


def load_documents() -> pl.DataFrame:
    """Load the four distinct domains in the canonical, grader-aligned order."""
    df = MLFPDataLoader().load("mlfp04", "sg_domain_qa.parquet")
    df = df.filter(pl.col("category").is_in(CATEGORIES)).sort(
        ["category", "instruction"]
    )
    return df.with_columns(
        (pl.col("instruction") + " " + pl.col("response")).alias("text")
    )


def build_tfidf(docs: list[str], *, min_df: int = 5, max_df_frac: float = 0.15):
    """Deterministic TF-IDF: sublinear TF, df-pruned vocab, L2-normalised rows."""
    tokens = [
        [t for t in re.findall(r"[a-z]{3,}", d.lower()) if t not in STOPWORDS]
        for d in docs
    ]
    n = len(tokens)
    dfreq = Counter()
    for ts in tokens:
        for w in set(ts):
            dfreq[w] += 1
    vocab = sorted(w for w, cnt in dfreq.items() if min_df <= cnt <= max_df_frac * n)
    vidx = {w: i for i, w in enumerate(vocab)}
    idf = np.array([np.log((1 + n) / (1 + dfreq[w])) + 1 for w in vocab])

    M = np.zeros((n, len(vocab)))
    for i, ts in enumerate(tokens):
        for w, cnt in Counter(ts).items():
            if w in vidx:
                M[i, vidx[w]] = 1 + np.log(cnt)

    M = M * idf
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms, vocab


def _purity(true: np.ndarray, pred: np.ndarray) -> float:
    return float(
        sum(
            np.unique(true[pred == c], return_counts=True)[1].max()
            for c in np.unique(pred)
        )
        / len(true)
    )


def solve() -> dict:
    """Discover four topics via TF-IDF + NMF — kailash-ml DimReductionEngine."""

    # Task 1: Load the documents and build the TF-IDF matrix
    frame = load_documents()
    M, _vocab = build_tfidf(frame["text"].to_list())

    # Task 2: Wrap the TF-IDF matrix in a Polars DataFrame and perform NMF
    matrix_df = pl.from_numpy(
        M,
        schema=[f"t{i}" for i in range(M.shape[1])]
    )

    engine = DimReductionEngine()

    result = engine.reduce(
        matrix_df,
        algorithm="nmf",
        n_components=N_TOPICS,
        seed=42,
    )

    # Task 3: Assign each document to the topic with the highest weight
    transformed = np.array(result.transformed)
    doc_topics = np.argmax(transformed, axis=1).tolist()

    # Task 4: Compute the topic purity
    categories = frame["category"].to_list()
    unique_categories = sorted(set(categories))
    category_map = {c: i for i, c in enumerate(unique_categories)}
    true_labels = np.array([category_map[c] for c in categories])

    topic_purity = _purity(true_labels, np.array(doc_topics))

    # Task 5: Return the required output
    return {
        "doc_topics": doc_topics,
        "n_topics": N_TOPICS,
        "topic_purity": float(topic_purity),
    }


if __name__ == "__main__":
    print(solve())