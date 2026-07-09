# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP05 — Assessment Task 1: Autoencoder Anomaly Detection

Complete the `solve()` function. Read problem.md for the full specification.

Train an UNDERCOMPLETE autoencoder on healthy-only sensor cycles, then score the
test set by reconstruction error. The grader checks the detector actually separates
the planted anomalies (ROC-AUC >= 0.90).

    python grader.py starter.py     # grade your attempt
    python grader.py solution.py    # verify the reference passes

No GPU required — trains on CPU in well under 15 seconds.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

INPUT_DIM = 12
SEED = 7


def make_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic synthetic sensor telemetry — DO NOT EDIT.

    Returns (X_train, X_test, y_test):
      X_train (800, 12) float32 — healthy cycles only.
      X_test  (400, 12) float32 — healthy + anomalous mix.
      y_test  (400,)    int     — 0 = healthy, 1 = anomaly (eval only).
    """
    rng = np.random.default_rng(SEED)
    # Healthy manifold: 3 latent factors projected into 12 channels + small noise.
    basis = rng.normal(size=(3, INPUT_DIM))

    def healthy(n: int) -> np.ndarray:
        z = rng.normal(size=(n, 3))
        return (z @ basis + 0.15 * rng.normal(size=(n, INPUT_DIM))).astype(np.float32)

    def anomaly(n: int) -> np.ndarray:
        # Off-manifold: independent per-channel signal that breaks the correlation.
        return (2.5 * rng.normal(size=(n, INPUT_DIM))).astype(np.float32)

    X_train = healthy(800)
    n_test_healthy, n_test_anom = 320, 80
    X_test = np.vstack([healthy(n_test_healthy), anomaly(n_test_anom)])
    y_test = np.concatenate(
        [np.zeros(n_test_healthy, dtype=int), np.ones(n_test_anom, dtype=int)]
    )
    perm = rng.permutation(len(y_test))
    return X_train, X_test[perm], y_test[perm]


def solve() -> dict:
    """Train an undercomplete AE on healthy data; return scored test set.

    See problem.md for the exact return contract.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    X_train, X_test, y_test = make_dataset()

    # TODO 1: choose a bottleneck size that is STRICTLY smaller than INPUT_DIM (12).
    #         A value around 3-5 captures the healthy manifold without copying input.
    latent_dim = 4  # <- replace with your undercomplete bottleneck (1..11)

    # TODO 2: build an undercomplete autoencoder as a torch.nn.Module.
    #         encoder: INPUT_DIM -> ... -> latent_dim
    #         decoder: latent_dim -> ... -> INPUT_DIM
    #         forward(x) should return the reconstruction (same shape as x).
    class AE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # self.encoder = nn.Sequential(...)
            self.encoder = nn.Sequential(
                nn.Linear(INPUT_DIM, 8),
                nn.ReLU(),
                nn.Linear(8, latent_dim),
            )

            # self.decoder = nn.Sequential(...)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 8),
                nn.ReLU(),
                nn.Linear(8, INPUT_DIM),
            )

        def forward(self, x):
            # return self.decoder(self.encoder(x))
            return self.decoder(self.encoder(x))  # <- replace

    model = AE()

    # TODO 3: train the AE with MSE reconstruction loss on X_train ONLY.
    #         ~40 epochs of Adam (lr=1e-3) on batches of healthy cycles is plenty.
    #         Hint: loss = F.mse_loss(model(batch), batch)
    # train_tensor = torch.tensor(X_train)
    # loader = DataLoader(TensorDataset(train_tensor), batch_size=64, shuffle=True)
    # optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    # for epoch in range(40): ...

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=64,
        shuffle=True,
    )

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(40):
        for (batch,) in loader:
            optimiser.zero_grad()

            reconstruction = model(batch)
            loss = F.mse_loss(reconstruction, batch)

            loss.backward()
            optimiser.step()

    # TODO 4: score the test set — per-row mean squared reconstruction error.
    #         model.eval(); no_grad; scores = ((X_test - recon) ** 2).mean(axis=1)

    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        recon = model(test_tensor)
        scores = ((test_tensor - recon) ** 2).mean(dim=1).cpu().numpy()

    return {
        "model": model,
        "scores": scores,
        "y_test": y_test,
        "input_dim": INPUT_DIM,
        "latent_dim": latent_dim,
    }


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score

    out = solve()
    auc = roc_auc_score(out["y_test"], out["scores"])
    s = out["scores"]
    yt = out["y_test"]
    sep = s[yt == 1].mean() / max(s[yt == 0].mean(), 1e-9)
    print(f"latent_dim={out['latent_dim']}  AUC={auc:.3f}  separation={sep:.2f}")