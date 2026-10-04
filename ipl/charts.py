"""Matplotlib figure builders (no pyplot state, so nothing leaks between reruns)."""
from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure

from ipl.features import match_table

ACCENT = "#ee0979"
DARK = "#1a1a2e"
GOOD = "#1a936f"


def _fig(w=7, h=4.5):
    fig = Figure(figsize=(w, h), layout="constrained")
    return fig, fig.subplots()


def venue_counts(df: pd.DataFrame, top: int = 10) -> Figure:
    counts = match_table(df)["venue"].value_counts().head(top).iloc[::-1]
    fig, ax = _fig()
    ax.barh(counts.index, counts.values, color=ACCENT)
    ax.set_title(f"Matches per Venue (Top {top})")
    ax.set_xlabel("Matches")
    return fig


def team_avg_totals(df: pd.DataFrame) -> Figure:
    mt = match_table(df)
    g = mt.groupby("bat_team")["total"].agg(["mean", "count"]).sort_values("mean")
    fig, ax = _fig()
    ax.barh([f"{t} (n={int(n)})" for t, n in zip(g.index, g["count"])], g["mean"], color=DARK)
    ax.set_title("Average First-Innings Total by Batting Team")
    ax.set_xlabel("Average total (runs, per match)")
    return fig


def season_avg_totals(df: pd.DataFrame) -> Figure:
    mt = match_table(df)
    g = mt.groupby(mt["date"].dt.year)["total"].mean()
    fig, ax = _fig()
    ax.plot(g.index, g.values, marker="o", color=ACCENT, linewidth=2)
    ax.set_xticks(list(g.index))
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Average First-Innings Total by Season")
    ax.set_ylabel("Runs")
    return fig


def totals_hist(df: pd.DataFrame) -> Figure:
    totals = match_table(df)["total"]
    fig, ax = _fig()
    ax.hist(totals, bins=25, color=ACCENT, edgecolor="white")
    ax.axvline(totals.mean(), color=DARK, linestyle="--", label=f"mean {totals.mean():.0f}")
    ax.set_title("Distribution of Final First-Innings Totals")
    ax.set_xlabel("Final score")
    ax.set_ylabel("Matches")
    ax.legend()
    return fig


def cv_chart(cv: pd.DataFrame) -> Figure:
    fig, ax = _fig(6, 4)
    ax.plot(cv["Fold"], cv["MAE"], marker="o", color=ACCENT, linewidth=2)
    ax.set_xticks(list(cv["Fold"]))
    ax.set_xlabel("Fold (chronological)")
    ax.set_ylabel("MAE (runs)")
    ax.set_title("XGBoost MAE by validation fold")
    return fig


def importance_chart(importances: pd.Series) -> Figure:
    fig, ax = _fig(8, 4.5)
    ax.barh(importances.index, importances.values, color=DARK)
    ax.set_title("XGBoost feature importance")
    return fig


def contribution_chart(contribs: pd.Series) -> Figure:
    order = contribs.abs().sort_values().index
    vals = contribs[order]
    fig, ax = _fig(8, 4.5)
    ax.barh(vals.index, vals.values, color=[ACCENT if v < 0 else GOOD for v in vals])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Impact on the XGBoost prediction (runs)")
    return fig
