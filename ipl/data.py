"""Loading, cleaning and input-validation helpers (pandas only)."""
from __future__ import annotations

import pandas as pd

from ipl.config import (
    BALLS_PER_INNINGS, DATA_PATH, TEAM_NAME_MAP, VENUE_NAME_MAP,
)

MAX_RUNS_PER_BALL = 7   # 6 off the bat + a no-ball extra; loose upper sanity bound
LAST5_WINDOW_BALLS = 30


# ---------------------------------------------------------------------------
# Overs <-> balls
# ---------------------------------------------------------------------------
def overs_to_balls(overs: float) -> int:
    """Dataset ``overs`` (e.g. 10.3 = 10 overs and 3 balls) -> balls bowled."""
    whole = int(overs)
    return whole * 6 + int(round((overs - whole) * 10))


def balls_to_dataset_overs(balls: int) -> float:
    """Balls bowled -> the ``overs`` value the dataset uses for that state.

    The raw data writes the last ball of an over as ``x.6`` (never ``x+1.0``),
    e.g. 18 balls bowled is ``2.6``. Using this keeps live inputs in the same
    representation the models were trained on.
    """
    if balls <= 0:
        return 0.0
    over, ball = divmod(balls - 1, 6)
    return over + (ball + 1) / 10


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def add_ball_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["balls_bowled"] = out["overs"].apply(overs_to_balls)
    out["balls_left"] = BALLS_PER_INNINGS - out["balls_bowled"]
    out["current_run_rate"] = (
        out["runs"] / out["balls_bowled"].where(out["balls_bowled"] > 0) * 6
    ).fillna(0.0)
    return out


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Read the CSV, harmonise team/venue names and sort chronologically."""
    df = pd.read_csv(path)
    df["_row"] = range(len(df))
    df["date"] = pd.to_datetime(df["date"])
    for col in ("bat_team", "bowl_team"):
        df[col] = df[col].replace(TEAM_NAME_MAP)
    df["venue"] = df["venue"].replace(VENUE_NAME_MAP)
    df = df.sort_values(["date", "mid", "_row"], kind="stable").reset_index(drop=True)
    return add_ball_features(df)


# ---------------------------------------------------------------------------
# Context helpers for the UI
# ---------------------------------------------------------------------------
def team_seasons(df: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """First and last season (year) each franchise appears as the batting side."""
    years = df.assign(year=df["date"].dt.year).groupby("bat_team")["year"].agg(["min", "max"])
    return {team: (int(r["min"]), int(r["max"])) for team, r in years.iterrows()}


def matchup_count(df: pd.DataFrame, team_a: str, team_b: str) -> int:
    """Number of matches in the data between the two teams (either batting order)."""
    matches = df.groupby("mid").agg(bat=("bat_team", "first"), bowl=("bowl_team", "first"))
    both = ((matches["bat"] == team_a) & (matches["bowl"] == team_b)) | (
        (matches["bat"] == team_b) & (matches["bowl"] == team_a)
    )
    return int(both.sum())


# ---------------------------------------------------------------------------
# Live-input handling
# ---------------------------------------------------------------------------
def is_innings_over(wickets: int, balls_bowled: int) -> bool:
    return wickets >= 10 or balls_bowled >= BALLS_PER_INNINGS


def effective_last5(balls_bowled, runs, wickets, runs_last_5, wickets_last_5):
    """Return ``(runs_last_5, wickets_last_5, adjusted)``.

    Within the first 5 overs the "last 5 overs" window is the whole innings so
    far, so the window stats are set to the running totals.
    """
    if balls_bowled <= LAST5_WINDOW_BALLS:
        adjusted = (runs_last_5, wickets_last_5) != (runs, wickets)
        return runs, wickets, adjusted
    return runs_last_5, wickets_last_5, False


def validate_match_state(
    runs: int, wickets: int, overs_completed: int, balls_in_over: int,
    runs_last_5: int, wickets_last_5: int,
) -> list[str]:
    """Return human-readable problems with a live match state (empty = valid)."""
    errors: list[str] = []
    if not 0 <= balls_in_over <= 5:
        errors.append("Balls in the current over must be between 0 and 5.")
    if overs_completed >= 20 and balls_in_over != 0:
        errors.append("An innings has at most 20 overs: set balls in the current over to 0.")
        return errors
    balls = overs_completed * 6 + balls_in_over
    if balls > BALLS_PER_INNINGS:
        errors.append("More than 120 balls bowled is not possible in a T20 innings.")
        return errors
    if runs > MAX_RUNS_PER_BALL * balls:
        errors.append(
            f"{runs} runs from {balls} balls is not plausible "
            f"(more than {MAX_RUNS_PER_BALL} runs per ball)."
        )
    if wickets > balls:
        errors.append("More wickets than balls bowled is not possible.")
    r5, w5, _ = effective_last5(balls, runs, wickets, runs_last_5, wickets_last_5)
    if r5 > runs:
        errors.append("Runs in the last 5 overs cannot exceed the current runs.")
    if w5 > wickets:
        errors.append("Wickets in the last 5 overs cannot exceed the current wickets.")
    return errors
