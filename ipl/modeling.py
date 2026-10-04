"""Training, persistence, validation and prediction for the XGBoost + NN ensemble.

xgboost / tensorflow are imported lazily so the rest of the package (and its
unit tests) work without them.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ipl import config as C
from ipl.data import balls_to_dataset_overs, load_data
from ipl.evaluation import (
    finalize_frame, live_mask, mae_by_stage, mae_rmse, residual_offsets_by_stage,
)
from ipl.features import (
    Split, build_split, chronological_split, match_table, split_recent_matches,
)


# ---------------------------------------------------------------------------
# Fingerprint: anything that changes what a saved model means changes this.
# ---------------------------------------------------------------------------
def compute_fingerprint(data_path: Path = C.DATA_PATH) -> str:
    cfg = {
        "model_version": C.MODEL_VERSION,
        "features": C.FEATURE_COLS,
        "xgb": C.XGB_PARAMS,
        "nn": C.NN_PARAMS,
        "nn_arch": C.NN_ARCH,
        "test_frac": C.TEST_FRAC,
        "val_frac": C.VAL_FRAC,
        "enc_smoothing": C.ENC_SMOOTHING,
        "teams": C.TEAM_NAME_MAP,
        "venues": C.VENUE_NAME_MAP,
    }
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode())
    h.update(Path(data_path).read_bytes())
    return h.hexdigest()[:10]


def _artifact(model_dir: Path, name: str, fp: str, ext: str) -> Path:
    return Path(model_dir) / f"{name}_{fp}.{ext}"


def _replace_atomically(tmp: Path, final: Path) -> None:
    os.replace(tmp, final)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
def make_xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(**C.XGB_PARAMS)


def train_or_load_xgb(X, y, fp: str, model_dir: Path = C.MODEL_DIR):
    path = _artifact(model_dir, "xgb", fp, "joblib")
    if path.exists():
        return joblib.load(path)
    model = make_xgb()
    model.fit(X, y)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    joblib.dump(model, tmp)
    _replace_atomically(tmp, path)
    return model


def xgb_contributions(model, row: pd.DataFrame) -> tuple[pd.Series, float]:
    """Exact TreeSHAP values from XGBoost itself: (per-feature impact, base value)."""
    import xgboost as xgb
    contribs = model.get_booster().predict(xgb.DMatrix(row[C.FEATURE_COLS]), pred_contribs=True)[0]
    return pd.Series(contribs[:-1], index=C.FEATURE_COLS), float(contribs[-1])


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------
def load_or_fit_scaler(X_train: pd.DataFrame, fp: str, model_dir: Path = C.MODEL_DIR) -> MinMaxScaler:
    path = _artifact(model_dir, "scaler", fp, "joblib")
    if path.exists():
        return joblib.load(path)
    scaler = MinMaxScaler().fit(X_train)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    joblib.dump(scaler, tmp)
    _replace_atomically(tmp, path)
    return scaler


def train_or_load_nn(fit_X, fit_y, val_X, val_y, fp: str, model_dir: Path = C.MODEL_DIR):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    path = _artifact(model_dir, "nn", fp, "keras")
    if path.exists():
        return tf.keras.models.load_model(path)

    tf.keras.utils.set_random_seed(C.RANDOM_STATE)
    (u1, u2), (d1, d2) = C.NN_ARCH["units"], C.NN_ARCH["dropout"]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(fit_X.shape[1],)),
        tf.keras.layers.Dense(u1, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(d1),
        tf.keras.layers.Dense(u2, activation="relu"),
        tf.keras.layers.Dropout(d2),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss=tf.keras.losses.Huber())
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=C.NN_PARAMS["patience"], restore_best_weights=True)
    model.fit(fit_X, fit_y, validation_data=(val_X, val_y),
              epochs=C.NN_PARAMS["epochs"], batch_size=C.NN_PARAMS["batch_size"],
              callbacks=[early_stop], verbose=0)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.keras")
    model.save(tmp)
    _replace_atomically(tmp, path)
    return model


# ---------------------------------------------------------------------------
# Expanding-window validation (XGBoost only; the NN is evaluated on the holdout)
# ---------------------------------------------------------------------------
def expanding_window_cv(df: pd.DataFrame, model_factory=make_xgb, folds: int = C.CV_FOLDS) -> pd.DataFrame:
    """Train on the first k blocks of matches, test on block k+1 (chronological)."""
    mt = match_table(df)
    block = len(mt) // (folds + 1)
    rows = []
    for k in range(1, folds + 1):
        tr, te = mt.iloc[: k * block], mt.iloc[k * block: (k + 1) * block]
        split = build_split(df, tr, te)
        model = model_factory()
        model.fit(split.X_train, split.y_train)
        mask = live_mask(split.X_test)
        pred = finalize_frame(model.predict(split.X_test), split.X_test)
        rows.append({
            "Fold": k, "Train matches": len(tr), "Test matches": len(te),
            "Test period": f"{te['date'].min():%Y-%m} to {te['date'].max():%Y-%m}",
            "MAE": mae_rmse(split.y_test[mask], pred[mask])["MAE"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Everything the app needs, built once
# ---------------------------------------------------------------------------
@dataclass
class AppState:
    fingerprint: str
    data: pd.DataFrame
    split: Split
    xgb_model: object
    nn_model: object
    scaler: MinMaxScaler
    metrics: dict
    stage_mae: pd.DataFrame
    offsets: dict
    cv_results: pd.DataFrame | None
    n_eval_rows: int
    importances: pd.Series = field(default=None)

    def predict_raw(self, X: pd.DataFrame) -> dict:
        xgb_p = np.asarray(self.xgb_model.predict(X), dtype=float)
        nn_p = np.asarray(self.nn_model.predict(self.scaler.transform(X), verbose=0), dtype=float).ravel()
        return {C.MODEL_NAMES[0]: xgb_p, C.MODEL_NAMES[1]: nn_p, C.MODEL_NAMES[2]: (xgb_p + nn_p) / 2}

    def predict_state(self, venue, bat_team, bowl_team, runs, wickets, balls_bowled,
                      runs_last_5, wickets_last_5):
        """-> (row, {model: finalized prediction}) for a live match state."""
        balls_left = C.BALLS_PER_INNINGS - balls_bowled
        row = pd.DataFrame([{
            "runs": runs, "wickets": wickets, "overs": balls_to_dataset_overs(balls_bowled),
            "runs_last_5": runs_last_5, "wickets_last_5": wickets_last_5,
            "balls_left": balls_left,
            "current_run_rate": runs / balls_bowled * 6 if balls_bowled > 0 else 0.0,
            **self.split.encodings.lookup(venue, bat_team, bowl_team),
        }])[C.FEATURE_COLS]
        raw = self.predict_raw(row)
        return row, {k: float(finalize_frame(v, row)[0]) for k, v in raw.items()}


def build_state(model_dir: Path = C.MODEL_DIR, data_path: Path = C.DATA_PATH,
                run_cv: bool = True) -> AppState:
    fp = compute_fingerprint(data_path)
    df = load_data(data_path)
    split = chronological_split(df)

    xgb_model = train_or_load_xgb(split.X_train, split.y_train, fp, model_dir)

    scaler = load_or_fit_scaler(split.X_train, fp, model_dir)
    fit_df, val_df = split_recent_matches(split.train, C.VAL_FRAC)
    nn_model = train_or_load_nn(
        scaler.transform(fit_df[C.FEATURE_COLS]), fit_df["total"],
        scaler.transform(val_df[C.FEATURE_COLS]), val_df["total"], fp, model_dir)

    state = AppState(fp, df, split, xgb_model, nn_model, scaler, {}, pd.DataFrame(), {}, None, 0)

    X_test, y_test = split.X_test, split.y_test
    mask = live_mask(X_test)
    raw = state.predict_raw(X_test)
    final = {k: finalize_frame(v, X_test) for k, v in raw.items()}
    state.n_eval_rows = int(mask.sum())
    state.metrics = {k: mae_rmse(y_test[mask], v[mask]) for k, v in final.items()}
    balls_left = X_test["balls_left"].to_numpy()
    state.stage_mae = pd.DataFrame({"Stage": mae_by_stage(y_test[mask], final[C.MODEL_NAMES[0]][mask], balls_left[mask])["Stage"]})
    for name, v in final.items():
        state.stage_mae[name] = mae_by_stage(y_test[mask], v[mask], balls_left[mask])["MAE"].to_numpy()
    state.stage_mae["Rows"] = mae_by_stage(y_test[mask], final[C.MODEL_NAMES[0]][mask], balls_left[mask])["Rows"]
    state.offsets = {k: residual_offsets_by_stage(y_test[mask], v[mask], balls_left[mask])
                     for k, v in final.items()}
    state.importances = pd.Series(xgb_model.feature_importances_, index=C.FEATURE_COLS).sort_values()
    if run_cv:
        state.cv_results = expanding_window_cv(df)
    return state
