# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP05 — Assessment Task 2: Tiny CNN for Image Classification

Complete the `solve()` function. Read problem.md for the full specification.

Build a convolutional neural network FROM SCRATCH and train it to classify bundled
8x8 handwritten digits. The grader re-runs your model and requires test accuracy
>= 0.90.

    python grader.py starter.py     # grade your attempt
    python grader.py solution.py    # verify the reference passes

No GPU required — trains on CPU in well under 25 seconds.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

N_CLASSES = 10
SEED = 42


def make_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic 8x8 digit split — DO NOT EDIT.

    Returns (X_train, y_train, X_test, y_test):
      X_* (N, 1, 8, 8) float32 in [0, 1];  y_* (N,) int 0..9.
    """
    digits = load_digits()
    X = (digits.images / 16.0).astype(np.float32)[:, None, :, :]  # (N, 1, 8, 8)
    y = digits.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    return X_train, y_train, X_test, y_test


def solve() -> dict:
    """Build + train a CNN from scratch; return predictions on the test split."""
    torch.manual_seed(SEED)
    X_train, y_train, X_test, y_test = make_dataset()

    # TODO 1: build a small CNN as a torch.nn.Module.
    #         It MUST contain at least one nn.Conv2d layer.
    #         A working recipe for 8x8 inputs:
    #           Conv2d(1->16, 3, padding=1) -> BatchNorm2d(16) -> ReLU -> MaxPool2d(2)  # 8->4
    #           Conv2d(16->32, 3, padding=1) -> BatchNorm2d(32) -> ReLU -> MaxPool2d(2) # 4->2
    #           Flatten -> Linear(32*2*2 -> 64) -> ReLU -> Linear(64 -> 10)
    class TinyCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 2 * 2, 64),
                nn.ReLU(),
                nn.Linear(64, N_CLASSES),
            )

        def forward(self, x):
            return self.head(self.features(x))

    model = TinyCNN()

    # TODO 2: count the nn.Conv2d layers you actually defined.
    n_conv = sum(isinstance(layer, nn.Conv2d) for layer in model.modules())

    # TODO 3: train with cross-entropy on (X_train, y_train).
    #         ~25 epochs of Adam (lr=1e-3), batch size 64 is enough.
    #         loss = F.cross_entropy(model(xb), yb)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
    )

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    model.train()

    for epoch in range(25):
        for xb, yb in loader:
            optimiser.zero_grad()

            outputs = model(xb)

            loss = F.cross_entropy(outputs, yb)

            loss.backward()

            optimiser.step()

    # TODO 4: predict on X_test (argmax of logits).

    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        outputs = model(X_test_tensor)

        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    return {
        "model": model,
        "preds": preds,
        "y_test": y_test,
        "n_conv": n_conv,
    }


if __name__ == "__main__":
    out = solve()
    acc = (out["preds"] == out["y_test"]).mean()
    print(f"conv_layers={out['n_conv']}  test_acc={acc:.3f}")