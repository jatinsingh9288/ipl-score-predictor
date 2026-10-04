"""Chronological splitting and leakage-safe target encodings (pandas only).

Venue / team strength is encoded as the (shrunken) mean final score. For
*training* rows that mean is built "ordered": a match only sees matches played
before it, so its own total never leaks into its own features. For test rows
and live predictions the encoding uses all training matches.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ipl.config import ENC_SMOOTHING, ENCODING_COLS, FEATURE_COLS, TEST_FRAC

_KEYS = (
    ("venue", "venue_avg_score"),
    ("bat_team", "bat_team_avg_score"),
    ("bowl_team", "bowl_team_avg_score"),
)


def match_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per match (its first-innings total), in chronological order."""
    mt = (
        df.groupby("mid")
        .agg(date=("date", "first"), venue=("venue", "first"),
             bat_team=("bat_team", "first"), bowl_team=("bowl_team", "first"),
             total=("total", "first"))
        .reset_index()
        .sort_values(["date", "mid"], kind="stable")
        .reset_index(drop=True)
    )
    return mt


def ordered_encodings(matches: pd.DataFrame, smoothing: float = ENC_SMOOTHING) -> pd.DataFrame:
    """Past-only, shrunken mean totals for each match (index = ``mid``).

    ``matches`` must be chronological. The very first match has no history, so
    its global prior is the mean of the *other* matches (never its own total).
    """
    m = matches.reset_index(drop=True)
    total = m["total"].astype(float)
    n_prev = np.arange(len(m))
    prior = (total.cumsum() - total) / np.maximum(n_prev, 1)
    if len(m) > 1:
        prior.iloc[0] = (total.sum() - total.iloc[0]) / (len(m) - 1)
    else:
        prior.iloc[0] = total.iloc[0]
    out = pd.DataFrame({"mid": m["mid"]})
    for key, col in _KEYS:
        g = m.groupby(key)["total"]
        prev_sum = g.cumsum() - total
        prev_n = g.cumcount()
        out[col] = (prev_sum + smoothing * prior) / (prev_n + smoothing)
    return out.set_index("mid")


@dataclass
class Encodings:
    """Full-history encodings used for test rows and live predictions."""
    global_avg: float
    venue_avg: dict
    bat_avg: dict
    bowl_avg: dict

    @classmethod
    def fit(cls, matches: pd.DataFrame, smoothing: float = ENC_SMOOTHING) -> "Encodings":
        glob = float(matches["total"].mean())

        def shrunk(key):
            g = matches.groupby(key)["total"].agg(["sum", "count"])
            return ((g["sum"] + smoothing * glob) / (g["count"] + smoothing)).to_dict()

        return cls(glob, shrunk("venue"), shrunk("bat_team"), shrunk("bowl_team"))

    def lookup(self, venue, bat_team, bowl_team) -> dict:
        return {
            "venue_avg_score": self.venue_avg.get(venue, self.global_avg),
            "bat_team_avg_score": self.bat_avg.get(bat_team, self.global_avg),
            "bowl_team_avg_score": self.bowl_avg.get(bowl_team, self.global_avg),
        }

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["venue_avg_score"] = out["venue"].map(self.venue_avg).fillna(self.global_avg)
        out["bat_team_avg_score"] = out["bat_team"].map(self.bat_avg).fillna(self.global_avg)
        out["bowl_team_avg_score"] = out["bowl_team"].map(self.bowl_avg).fillna(self.global_avg)
        return out


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    encodings: Encodings
    train_matches: pd.DataFrame
    test_matches: pd.DataFrame

    @property
    def X_train(self): return self.train[FEATURE_COLS]
    @property
    def y_train(self): return self.train["total"]
    @property
    def X_test(self): return self.test[FEATURE_COLS]
    @property
    def y_test(self): return self.test["total"]


def build_split(df: pd.DataFrame, train_matches: pd.DataFrame, test_matches: pd.DataFrame) -> Split:
    """Rows of ``df`` for the given matches, with leakage-safe encodings attached."""
    train = df[df["mid"].isin(train_matches["mid"])]
    test = df[df["mid"].isin(test_matches["mid"])]
    ordered = ordered_encodings(train_matches)
    train = train.drop(columns=ENCODING_COLS, errors="ignore").join(ordered, on="mid")
    enc = Encodings.fit(train_matches)
    test = enc.apply(test.drop(columns=ENCODING_COLS, errors="ignore"))
    return Split(train.reset_index(drop=True), test.reset_index(drop=True), enc,
                 train_matches, test_matches)


def chronological_split(df: pd.DataFrame, test_frac: float = TEST_FRAC) -> Split:
    """Oldest ``1 - test_frac`` of matches train, the most recent ones test."""
    mt = match_table(df)
    n_train = int(len(mt) * (1 - test_frac))
    return build_split(df, mt.iloc[:n_train], mt.iloc[n_train:])


def split_recent_matches(train: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the most recent ``frac`` of *matches* (not rows) for early stopping."""
    mt = match_table(train)
    n_val = max(1, int(round(len(mt) * frac)))
    val_mids = set(mt["mid"].iloc[-n_val:])
    is_val = train["mid"].isin(val_mids)
    return train[~is_val], train[is_val]
