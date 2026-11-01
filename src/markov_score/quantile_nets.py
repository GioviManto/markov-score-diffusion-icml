"""Quantile networks: the standard deep probabilistic forecasters E59 compares against.

Each reads the last `window` past values and outputs, for every horizon, the 19 quantiles 5% ... 95% of the future
value, trained with the pinball (quantile) loss, the loss whose minimiser is the conditional quantile. Two architectures:
an MLP on the flattened window, and a GRU reading the window in order (the recurrent forecaster, as in MQ-RNN and the
encoders of DeepAR-type models). Quantile crossing is repaired by sorting at prediction time, as is standard.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn


class QuantileMLP(nn.Module):
    def __init__(self, window: int, n_horizons: int, n_levels: int, hidden=(128, 128)):
        super().__init__()
        layers, d = [], window
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers.append(nn.Linear(d, n_horizons * n_levels))
        self.net = nn.Sequential(*layers)
        self.shape = (n_horizons, n_levels)

    def forward(self, x):
        return self.net(x).view(-1, *self.shape)


class QuantileGRU(nn.Module):
    def __init__(self, window: int, n_horizons: int, n_levels: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_horizons * n_levels)
        self.shape = (n_horizons, n_levels)

    def forward(self, x):
        _, h = self.gru(x.unsqueeze(-1))
        return self.head(h[-1]).view(-1, *self.shape)


def build(arch: str, window: int, n_horizons: int, n_levels: int) -> nn.Module:
    if arch == "qmlp":
        return QuantileMLP(window, n_horizons, n_levels)
    if arch == "qgru":
        return QuantileGRU(window, n_horizons, n_levels)
    raise ValueError(f"unknown quantile network {arch!r}")


def pinball(pred: torch.Tensor, y: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Mean pinball loss; pred (B, H, Q), y (B, H)."""
    d = y.unsqueeze(-1) - pred
    return torch.maximum(levels * d, (levels - 1.0) * d).mean()


def train(arch, X, Y, Xv, Yv, levels, steps, checkpoints, lr=1e-3, batch=256, seed=0):
    """Adam on the pinball loss; the checkpoint with the lowest validation pinball is returned with its curve."""
    torch.manual_seed(int(seed))
    model = build(arch, X.shape[1], Y.shape[1], len(levels)).double()
    lv = torch.as_tensor(levels, dtype=torch.float64)
    Xt, Yt = torch.as_tensor(X, dtype=torch.float64), torch.as_tensor(Y, dtype=torch.float64)
    Xvt, Yvt = torch.as_tensor(Xv, dtype=torch.float64), torch.as_tensor(Yv, dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(int(seed))
    best, best_v, curve = None, np.inf, {}
    for step in range(1, max(checkpoints) + 1):
        idx = torch.randint(0, len(Xt), (min(batch, len(Xt)),), generator=gen)
        opt.zero_grad(set_to_none=True)
        pinball(model(Xt[idx]), Yt[idx], lv).backward()
        opt.step()
        if step in checkpoints:
            with torch.no_grad():
                v = float(pinball(model(Xvt), Yvt, lv))
            curve[step] = v
            if v < best_v:
                best, best_v = copy.deepcopy(model.state_dict()), v
    model.load_state_dict(best)
    n_params = int(sum(p.numel() for p in model.parameters()))
    return model, {"checkpoint": min(curve, key=curve.get), "val_pinball": best_v, "n_params": n_params,
                   "val_curve": ";".join(f"{k}:{v:.6g}" for k, v in sorted(curve.items()))}


def predict(model, X) -> np.ndarray:
    """(B, H, Q) sorted quantiles."""
    with torch.no_grad():
        q = model(torch.as_tensor(X, dtype=torch.float64)).numpy()
    return np.sort(q, axis=-1)
