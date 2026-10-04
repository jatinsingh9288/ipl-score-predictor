"""Metrics, stage-aware prediction ranges and post-processing (numpy/pandas only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ipl.config import (
    BALLS_PER_INNINGS, INTERVAL_QUANTILES, STAGE_LABELS, STAGE_SIZE_BALLS,
)


def stage_index(balls_left) -> np.ndarray:
    """0..3 = overs 0-5, 5-10, 10-15, 15-20 (by balls bowled)."""
    balls_bowled = BALLS_PER_INNINGS - np.asarray(balls_left)
    return np.clip(balls_bowled // STAGE_SIZE_BALLS, 0, len(STAGE_LABELS) - 1).astype(int)


def live_mask(X: pd.DataFrame) -> np.ndarray:
    """Rows where the innings is still in progress (the only ones worth predicting)."""
    return ((X["balls_left"] > 0) & (X["wickets"] < 10)).to_numpy()


def finalize(pred, runs, balls_left, wickets):
    """Floor at the current score; finished innings score exactly what they have."""
    pred = np.maximum(np.asarray(pred, dtype=float), np.asarray(runs, dtype=float))
    over = (np.asarray(balls_left) <= 0) | (np.asarray(wickets) >= 10)
    return np.where(over, np.asarray(runs, dtype=float), pred)


def finalize_frame(pred, X: pd.DataFrame):
    return finalize(pred, X["runs"], X["balls_left"], X["wickets"])


def mae_rmse(y, pred) -> dict:
    err = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2)))}


def mae_by_stage(y, pred, balls_left) -> pd.DataFrame:
    stages = stage_index(balls_left)
    err = np.abs(np.asarray(y, dtype=float) - np.asarray(pred, dtype=float))
    rows = []
    for i, label in enumerate(STAGE_LABELS):
        sel = stages == i
        rows.append({"Stage": label, "MAE": float(err[sel].mean()) if sel.any() else float("nan"),
                     "Rows": int(sel.sum())})
    return pd.DataFrame(rows)


def residual_offsets_by_stage(y, pred, balls_left, quantiles=INTERVAL_QUANTILES) -> dict:
    """Empirical residual quantiles (y - pred) per innings stage -> {stage: (lo, hi)}."""
    stages = stage_index(balls_left)
    resid = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    lo_q, hi_q = quantiles
    out = {}
    for i in range(len(STAGE_LABELS)):
        r = resid[stages == i]
        out[i] = (float(np.quantile(r, lo_q)), float(np.quantile(r, hi_q))) if len(r) else (0.0, 0.0)
    return out


def prediction_range(pred: float, runs: int, stage: int, offsets: dict) -> tuple[float, float]:
    lo, hi = offsets[stage]
    low = max(pred + lo, float(runs))
    return low, max(pred + hi, low)


def coverage(y, pred, balls_left, offsets) -> float:
    """Share of rows whose true value falls inside the stage-wise range."""
    stages = stage_index(balls_left)
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    lo = np.array([offsets[s][0] for s in stages])
    hi = np.array([offsets[s][1] for s in stages])
    return float(np.mean((y >= pred + lo) & (y <= pred + hi)))
