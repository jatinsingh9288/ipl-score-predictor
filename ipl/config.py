"""Central configuration: paths, name mappings, features and model settings.

Anything that changes what a trained model *means* (features, hyper-parameters,
name mappings, split fractions, ...) is hashed into the model fingerprint in
``ipl.modeling.compute_fingerprint`` so stale saved models are never reused.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ipl_data.csv"
MODEL_DIR = ROOT / "trained_models"

# Bump when feature / training *logic* changes in a way the settings below
# do not capture. Forces saved models to be rebuilt.
MODEL_VERSION = "2"

RANDOM_STATE = 42
BALLS_PER_INNINGS = 120
TEST_FRAC = 0.20          # most recent 20% of matches are the final holdout
VAL_FRAC = 0.15           # most recent 15% of *training matches* for NN early stopping
CV_FOLDS = 4
ENC_SMOOTHING = 5.0       # pseudo-matches of shrinkage towards the global mean
INTERVAL_QUANTILES = (0.10, 0.90)   # -> an 80% empirical range

# ---------------------------------------------------------------------------
# Name mappings
# ---------------------------------------------------------------------------
# Only genuine renames of the SAME franchise are merged.
#   Delhi Daredevils            -> Delhi Capitals               (rebranded 2019)
#   Kings XI Punjab             -> Punjab Kings                 (rebranded 2021)
#   Royal Challengers Bangalore -> Royal Challengers Bengaluru  (renamed 2024)
#   Rising Pune Supergiant      -> Rising Pune Supergiants      (inconsistent spelling in raw data)
# NOT merged (different franchises): Deccan Chargers vs Sunrisers Hyderabad,
# Gujarat Lions vs Gujarat Titans (the Titans only started in 2022).
TEAM_NAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}

# Same ground under two names; the date ranges in the data never overlap
# (Mohali: 2008-2015 vs 2016-2017, Pune: 2012-2013 vs 2015-2017).
VENUE_NAME_MAP = {
    "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "Subrata Roy Sahara Stadium": "Maharashtra Cricket Association Stadium",
}

TEAM_COLORS = {
    "Chennai Super Kings": "#FDB913",
    "Mumbai Indians": "#004BA0",
    "Royal Challengers Bengaluru": "#EC1C24",
    "Kolkata Knight Riders": "#3A225D",
    "Sunrisers Hyderabad": "#FF822A",
    "Delhi Capitals": "#17449B",
    "Punjab Kings": "#ED1B24",
    "Rajasthan Royals": "#EA1A85",
}
DEFAULT_TEAM_COLOR = "#ee0979"

# ---------------------------------------------------------------------------
# Features and models
# ---------------------------------------------------------------------------
ENCODING_COLS = ["venue_avg_score", "bat_team_avg_score", "bowl_team_avg_score"]
FEATURE_COLS = [
    "runs", "wickets", "overs", "runs_last_5", "wickets_last_5",
    "balls_left", "current_run_rate",
] + ENCODING_COLS

XGB_PARAMS = dict(
    n_estimators=150, max_depth=5, learning_rate=0.08,
    subsample=0.9, colsample_bytree=0.9, n_jobs=-1, random_state=RANDOM_STATE,
)
NN_PARAMS = dict(epochs=25, batch_size=256, patience=4)

MODEL_NAMES = ("XGBoost", "Neural Network", "Ensemble (avg)")

# Innings stages used for error reporting and prediction ranges (balls bowled).
STAGE_SIZE_BALLS = 30
STAGE_LABELS = ["Overs 0-5", "Overs 5-10", "Overs 10-15", "Overs 15-20"]

# Neural-network architecture (Dense -> BatchNorm -> Dropout -> Dense -> Dropout -> Dense(1))
NN_ARCH = dict(units=(128, 64), dropout=(0.2, 0.1))
