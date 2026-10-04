import numpy as np
import pandas as pd


def synthetic_matches(n=60, seed=0):
    """Small ball-by-ball-like frame: n matches, 12 balls each, chronological dates."""
    rng = np.random.default_rng(seed)
    venues = ["V1", "V2", "V3"]
    teams = ["A", "B", "C", "D"]
    rows = []
    for i in range(n):
        total = int(rng.integers(110, 220))
        bat, bowl = rng.choice(teams, 2, replace=False)
        for b in range(1, 13):
            balls = b * 10
            runs = int(total * balls / 120)
            rows.append(dict(
                mid=i + 1, date=pd.Timestamp("2010-01-01") + pd.Timedelta(days=3 * i),
                venue=venues[i % 3], bat_team=bat, bowl_team=bowl, total=total, runs=runs,
                wickets=int(rng.integers(0, 6)), overs=(balls - 1) // 6 + ((balls - 1) % 6 + 1) / 10,
                runs_last_5=int(runs * 0.4), wickets_last_5=0, _row=len(rows),
            ))
    from ipl.data import add_ball_features
    return add_ball_features(pd.DataFrame(rows))
