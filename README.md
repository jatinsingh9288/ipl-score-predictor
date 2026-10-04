# 🏏 IPL Score Predictor

Predicts the **final first-innings total** of an IPL match from any point in the innings, using an
**XGBoost + neural-network ensemble**, served through a **Streamlit** app.

> **Scope:** the dataset is ball-by-ball first-innings data from **IPL 2008–2017** (617 matches,
> 76,014 deliveries). There is no post-2017 data, so newer players, grounds and scoring trends are
> not represented. Treat this as a modelling project, not a tool for current matches.

## Features

- **Predict a score** for a live match state (venue, teams, runs, wickets, overs and balls, last-5-overs form),
  with the XGBoost, neural-network or averaged ensemble prediction shown side by side.
- **Input validation** (impossible overs/balls, last-5 stats larger than the totals, finished innings, …).
- **Stage-aware prediction range**: the 10th–90th percentile of holdout errors *for that stage of the innings*
  (early-over predictions are much less certain than death-over ones), never below the current score.
- **Explanation**: exact TreeSHAP values computed by XGBoost, clearly labelled as explaining the XGBoost component.
- **Validation tab**: expanding-window (chronological) cross-validation, final holdout comparison, and holdout
  MAE broken down by stage of the innings. All numbers are computed when you run the app; none are hard-coded.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # Python 3.10 or 3.11 recommended (TensorFlow compatibility)

python train.py                    # optional: trains/loads the models and prints the metrics
streamlit run main.py
```

The first run trains both models (a few minutes, CPU) and saves them to `trained_models/`;
later runs load them instantly. Artifacts are keyed by a **fingerprint** of the data file, features,
hyper-parameters, name mappings and split settings, so changing any of those retrains automatically
instead of silently loading a stale model. `python train.py --force` rebuilds from scratch.

## How it works

| Step | What happens |
|------|--------------|
| Data | Ball-by-ball rows; `overs` converted to `balls_bowled` / `balls_left` and a true current run rate. Sorted by date. |
| Features | `runs`, `wickets`, `overs`, `runs_last_5`, `wickets_last_5`, `balls_left`, `current_run_rate` + venue / batting-team / bowling-team average scores. |
| Split | Chronological: the oldest 80% of **matches** train, the most recent 20% test. |
| Leakage control | Venue/team average scores are shrunken, match-weighted means. For training rows they are *ordered*: a match only sees matches played **before** it (its own total never reaches its own features). Test rows and live predictions use all training matches. Scaler is fit on training data only and saved with the models. |
| XGBoost | `XGBRegressor`, parameters in `ipl/config.py`. |
| Neural network | Dense(128)–BatchNorm–Dropout–Dense(64)–Dropout–Dense(1), Huber loss, early stopping on the most recent 15% of *training matches*. |
| Ensemble | Simple average of the two (weights not tuned). |
| Post-processing | A prediction is never below the current score; a finished innings (10 wickets or 120 balls) is exactly the current score. |
| Evaluation | Only in-progress deliveries are scored. Deliveries within a match are strongly correlated, so the effective test size is ~124 matches, not ~15k rows. |

## Team and venue names

Only genuine renames of the **same franchise** are merged:

| Raw name | Used as |
|----------|---------|
| Delhi Daredevils | Delhi Capitals |
| Kings XI Punjab | Punjab Kings |
| Royal Challengers Bangalore | Royal Challengers Bengaluru |
| Rising Pune Supergiant | Rising Pune Supergiants |

**Not** merged because they are different franchises: Deccan Chargers vs Sunrisers Hyderabad, and
Gujarat Lions vs Gujarat Titans (the Titans only started in 2022). Two grounds that appear under two names are
merged (Mohali; the Maharashtra Cricket Association Stadium, formerly Subrata Roy Sahara Stadium) — their date
ranges in the data never overlap. All mappings live in `ipl/config.py`.

## Project structure

```
main.py              Streamlit app
train.py             train/load models, print metrics
ipl/config.py        paths, name mappings, features, hyper-parameters
ipl/data.py          loading, overs helpers, live-input validation
ipl/features.py      chronological split, leakage-safe encodings
ipl/evaluation.py    metrics, per-stage errors, prediction ranges, post-processing
ipl/modeling.py      fingerprint, training/loading, validation, prediction
ipl/charts.py        matplotlib figures
data/ipl_data.csv    dataset
tests/               unit tests
trained_models/      generated artifacts (git-ignored)
```

## Tests

```bash
python -m unittest discover -s tests -t .
```

The tests cover the overs conversion (round-trip for every ball), input validation, name mappings, the chronological
split, the absence of target leakage in the encodings, per-stage error and range logic, fingerprint invalidation,
scaler persistence, the expanding-window validation and chart construction. They run without TensorFlow or XGBoost
(scikit-learn stands in for the model-training logic).

## Known limitations

- Data ends in 2017; defunct teams (Deccan Chargers, Kochi Tuskers Kerala, Pune Warriors, Gujarat Lions, Rising Pune
  Supergiants) can be selected, and matchups that never happened in the data are extrapolations.
- First innings only; no player-level features, pitch/weather/toss information.
- 455 raw rows write the first ball of an over as `x.0` instead of `x.1`, and 3 matches contain a few out-of-order
  deliveries (28 decreasing steps in total); these are left as they are.
- The ensemble weights and the target-encoding shrinkage (5 pseudo-matches) are fixed choices, not tuned.
- The prediction range is estimated from the same holdout matches used for the reported metrics, so treat it as
  approximate.
- Cross-validation covers XGBoost only; the neural network is evaluated on the final holdout.

## Data

Ball-by-ball IPL first-innings records, 2008–2017 (`data/ipl_data.csv`). Columns used: `mid`, `date`, `venue`,
`bat_team`, `bowl_team`, `runs`, `wickets`, `overs`, `runs_last_5`, `wickets_last_5`, `total`. Add the original
source and its licence here before publishing.
