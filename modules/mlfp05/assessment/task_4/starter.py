# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP05 — Assessment Task 4: Tiny Transformer Text Classification

Complete the `solve()` function. Read problem.md for the full specification.

Build a tiny Transformer encoder FROM SCRATCH (token embeddings + self-attention +
mean-pool + linear) and train it to classify AG News headlines into 4 desks. The
grader re-runs your model and requires test accuracy >= 0.72.

    python grader.py starter.py     # grade your attempt
    python grader.py solution.py    # verify the reference passes

No GPU required — trains on CPU in well under 35 seconds.
"""
from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from shared import MLFPDataLoader

MAX_LEN = 40
MAX_VOCAB = 8000
N_CLASSES = 4
SEED = 5
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def make_dataset():
    """Deterministic AG News encoding — DO NOT EDIT.

    Returns (X_train, y_train, X_test, y_test, vocab_size):
      X_* (N, MAX_LEN) int64 — padded token indices (0=PAD, 1=UNK).
      y_* (N,) int64         — class labels 0..3.
    """
    train_df = MLFPDataLoader().load("mlfp05", "ag_news.parquet")
    test_df = MLFPDataLoader().load("mlfp05", "ag_news_test.parquet")
    train_texts = train_df["text"].to_list()
    test_texts = test_df["text"].to_list()
    y_train = train_df["label"].to_numpy().astype(np.int64)
    y_test = test_df["label"].to_numpy().astype(np.int64)

    # Vocab from TRAINING text only, by descending frequency (deterministic).
    from collections import Counter

    counts: Counter = Counter()
    for t in train_texts:
        counts.update(_tokenize(t))
    # 0 = PAD, 1 = UNK; most-common tokens fill the rest.
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, _ in counts.most_common(MAX_VOCAB - 2):
        vocab[tok] = len(vocab)

    def encode(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
        for i, t in enumerate(texts):
            toks = _tokenize(t)[:MAX_LEN]
            for j, tok in enumerate(toks):
                out[i, j] = vocab.get(tok, 1)
        return out

    X_train = encode(train_texts)
    X_test = encode(test_texts)
    return X_train, y_train, X_test, y_test, len(vocab)


def solve() -> dict:
    """Build + train a tiny transformer; return predictions on the test split."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    X_train, y_train, X_test, y_test, vocab_size = make_dataset()

    # TODO 1: build a tiny transformer encoder as a torch.nn.Module.
    #         It MUST use self-attention — use nn.MultiheadAttention OR
    #         nn.TransformerEncoderLayer / nn.TransformerEncoder.
    #         Recipe:
    #           nn.Embedding(vocab_size, dim, padding_idx=0)  + a positional param
    #           nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, heads,
    #               dim_feedforward=dim*2, dropout=0.1, batch_first=True),
    #               num_layers=2, enable_nested_tensor=False)
    #           mean-pool over non-pad tokens -> nn.Linear(dim, N_CLASSES)
    #         Pass src_key_padding_mask=(x == 0) so padding is ignored.
    class TinyTransformer(nn.Module):
        def __init__(self, vocab: int, dim: int = 96, heads: int = 4) -> None:
            super().__init__()

            # self.embed = nn.Embedding(vocab, dim, padding_idx=0)
            self.embed = nn.Embedding(vocab, dim, padding_idx=0)

            # self.pos = nn.Parameter(torch.zeros(1, MAX_LEN, dim))
            self.pos = nn.Parameter(torch.zeros(1, MAX_LEN, dim))

            # self.encoder = nn.TransformerEncoder(...)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 2,
                dropout=0.1,
                batch_first=True,
            )

            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=2,
                enable_nested_tensor=False,
            )

            # self.head = nn.Linear(dim, N_CLASSES)
            self.head = nn.Linear(dim, N_CLASSES)

        def forward(self, x):
            # pad_mask = x == 0
            pad_mask = x == 0

            # h = self.embed(x) + self.pos
            h = self.embed(x) + self.pos

            # h = self.encoder(h, src_key_padding_mask=pad_mask)
            h = self.encoder(h, src_key_padding_mask=pad_mask)

            # mask = (~pad_mask).unsqueeze(-1).float()
            mask = (~pad_mask).unsqueeze(-1).float()

            # pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)

            # return self.head(pooled)
            return self.head(pooled)

    model = TinyTransformer(vocab_size)

    # TODO 2: report whether the model uses self-attention (it must).
    uses_attention = True  # <- replace with True once you add the encoder

    # TODO 3: train with cross-entropy on (X_train, y_train).
    #         ~12 epochs of Adam (lr=1e-3), batch size 64 works well.
    #         loss = F.cross_entropy(model(xb), yb)
    # train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    # loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    # optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    # for epoch in range(12): ...

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.long),
    )

    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(12):
        for xb, yb in loader:
            optimiser.zero_grad()

            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            loss.backward()
            optimiser.step()

    # TODO 4: predict on X_test (argmax of logits).

    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.long)
        logits = model(test_tensor)
        preds = logits.argmax(dim=1).cpu().numpy()

    return {
        "model": model,
        "preds": preds,
        "y_test": y_test,
        "uses_attention": uses_attention,
    }


if __name__ == "__main__":
    out = solve()
    acc = (out["preds"] == out["y_test"]).mean()
    yt = out["y_test"]
    majority = np.bincount(yt).max() / len(yt)
    print(f"transformer  test_acc={acc:.2f}  majority={majority:.2f}")