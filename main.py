"""IPL Score Predictor - Streamlit app.   Run with:  streamlit run main.py"""
import streamlit as st

from ipl import charts
from ipl import config as C
from ipl.data import (
    effective_last5, is_innings_over, matchup_count, team_seasons, validate_match_state,
)
from ipl.evaluation import prediction_range, stage_index
from ipl.modeling import build_state, xgb_contributions

st.set_page_config(page_title="IPL Score Predictor", layout="wide", page_icon="🏏")

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px; font-weight: 600; padding: 10px 20px;
        background-color: #f0f2f6; border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] { background-color: #fff2e0; color: #d84315; }
    div[data-testid="stMetricValue"] { font-size: 30px; color: #1a1a2e; }
    div[data-testid="stMetric"] {
        background-color: #f7f8fc; padding: 14px 16px; border-radius: 10px;
        border: 1px solid #e6e8f0;
    }
    div.stButton > button { width: 100%; }
    h1 { color: #1a1a2e; }
    .score-banner {
        padding: 22px; border-radius: 14px; color: white; text-align: center;
        margin-top: 10px; margin-bottom: 10px;
    }
    .score-banner h2 { color: white; margin: 0; font-size: 42px; }
    .score-banner p { margin: 4px 0 0 0; opacity: 0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_state():
    """Load or train everything once per server process (artifacts are cached on disk)."""
    return build_state()


st.title("🏏 IPL Score Predictor")
st.caption(
    "Predicts the final first-innings total from any point in a match. Trained on "
    "ball-by-ball IPL data from **2008-2017**, evaluated chronologically (older matches "
    "train, the most recent ones test)."
)

with st.spinner("Loading models (the first run trains them and can take a few minutes)..."):
    state = get_state()

data = state.data
matches = data["mid"].nunique()
years = data["date"].dt.year
seasons = team_seasons(data)

tab_overview, tab_validation, tab_predict = st.tabs(
    ["📊 Overview", "🧪 Model Validation", "🎯 Predict a Score"])

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Matches", f"{matches}")
    c2.metric("Seasons", f"{years.min()}-{years.max()}")
    c3.metric("Venues", f"{data['venue'].nunique()}")
    c4.metric("Franchises", f"{data['bat_team'].nunique()}")
    c5.metric("Deliveries", f"{len(data):,}")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(charts.venue_counts(data))
    with col2:
        st.pyplot(charts.team_avg_totals(data))
    col3, col4 = st.columns(2)
    with col3:
        st.pyplot(charts.season_avg_totals(data))
    with col4:
        st.pyplot(charts.totals_hist(data))

    with st.expander("About the data and its limits"):
        st.markdown(
            f"""
- The dataset covers **{matches} first innings from {years.min()} to {years.max()}**. Nothing later than
  2017 is included, so newer players, teams, grounds and scoring trends are not represented.
- Team names use the current name where it is the **same franchise**: Delhi Daredevils → Delhi Capitals,
  Kings XI Punjab → Punjab Kings, Royal Challengers Bangalore → Bengaluru. Deccan Chargers and
  Sunrisers Hyderabad, and Gujarat Lions and (the 2022+) Gujarat Titans, are different franchises and
  are **not** merged.
- Two grounds that appear under two names are merged (Mohali; Pune's Gahunje stadium).
- Defunct teams (Deccan Chargers, Kochi Tuskers Kerala, Pune Warriors, Gujarat Lions, Rising Pune
  Supergiants) can be selected; predictions involving teams that never met are extrapolations.
            """
        )

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
with tab_validation:
    if state.cv_results is not None:
        st.subheader("Expanding-window validation (XGBoost)")
        st.write(
            "The model is retrained on progressively larger blocks of match history and "
            "tested only on the block of matches that follows it, so it never sees the future."
        )
        cv_col1, cv_col2 = st.columns([1.2, 1])
        with cv_col1:
            st.dataframe(state.cv_results.set_index("Fold").style.format({"MAE": "{:.2f}"}))
        with cv_col2:
            st.pyplot(charts.cv_chart(state.cv_results))
        st.info(
            f"Mean MAE across folds: **{state.cv_results['MAE'].mean():.2f} runs** "
            f"(std {state.cv_results['MAE'].std():.2f}). Scored on in-progress states only."
        )

    st.subheader("Final holdout: model comparison")
    sp = state.split
    st.caption(
        f"Trained on the oldest {len(sp.train_matches)} matches, tested on the most recent "
        f"{len(sp.test_matches)} ({sp.test_matches['date'].min():%b %Y} to "
        f"{sp.test_matches['date'].max():%b %Y}). {state.n_eval_rows:,} in-progress deliveries "
        "are scored; deliveries are correlated within a match."
    )
    cols = st.columns(len(state.metrics))
    for col, (name, m) in zip(cols, state.metrics.items()):
        col.metric(name, f"{m['MAE']:.1f} MAE", f"RMSE {m['RMSE']:.1f}", delta_color="off")

    st.markdown("**Holdout MAE by stage of the innings** (a single overall MAE hides the fact "
                "that early-innings predictions are much harder than late ones)")
    st.dataframe(state.stage_mae.set_index("Stage").style.format(
        {c: "{:.2f}" for c in state.stage_mae.columns if c not in ("Stage", "Rows")}))

    st.subheader("What drives the prediction?")
    st.pyplot(charts.importance_chart(state.importances))

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
with tab_predict:
    model_choice = st.radio("Model", list(state.metrics.keys()), horizontal=True)

    venues = sorted(data["venue"].unique())
    bat_teams = sorted(data["bat_team"].unique())

    c1, c2, c3 = st.columns(3)
    with c1:
        venue = st.selectbox("Venue", venues)
    with c2:
        bat_team = st.selectbox("Batting Team", bat_teams)
    with c3:
        bowl_team = st.selectbox("Bowling Team", [t for t in bat_teams if t != bat_team])

    lo1, hi1 = seasons[bat_team]
    lo2, hi2 = seasons[bowl_team] if bowl_team in seasons else (None, None)
    seen = matchup_count(data, bat_team, bowl_team)
    st.caption(
        f"{bat_team}: in the data {lo1}-{hi1} · {bowl_team}: "
        f"{f'{lo2}-{hi2}' if lo2 else 'bowling only'} · these two met {seen} time(s) in 2008-2017."
    )

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        runs = st.number_input("Current Runs", min_value=0, max_value=300, step=1, value=80)
    with c5:
        wickets = st.number_input("Current Wickets", min_value=0, max_value=10, step=1, value=2)
    with c6:
        overs_completed = st.number_input("Overs Completed", min_value=0, max_value=20, step=1, value=10)
    with c7:
        balls_in_over = st.number_input("Balls in Current Over", min_value=0, max_value=5, step=1, value=0)

    c8, c9 = st.columns(2)
    with c8:
        runs_last_5 = st.number_input("Runs in Last 5 Overs", min_value=0, max_value=120, step=1, value=40)
    with c9:
        wickets_last_5 = st.number_input("Wickets in Last 5 Overs", min_value=0, max_value=10, step=1, value=1)

    if st.button("Predict Score", type="primary"):
        errors = validate_match_state(runs, wickets, overs_completed, balls_in_over,
                                      runs_last_5, wickets_last_5)
        balls_bowled = overs_completed * 6 + balls_in_over
        if errors:
            for msg in errors:
                st.error(msg)
        elif is_innings_over(wickets, balls_bowled):
            st.success(f"The innings is over: the final score is **{runs}**.")
        else:
            runs_last_5, wickets_last_5, adjusted = effective_last5(
                balls_bowled, runs, wickets, runs_last_5, wickets_last_5)
            if adjusted:
                st.info("Within the first 5 overs the 'last 5 overs' window is the whole innings, "
                        "so those two inputs were set to the current runs and wickets.")

            row, preds = state.predict_state(venue, bat_team, bowl_team, runs, wickets,
                                             balls_bowled, runs_last_5, wickets_last_5)
            point = preds[model_choice]
            stage = int(stage_index([C.BALLS_PER_INNINGS - balls_bowled])[0])
            low, high = prediction_range(point, runs, stage, state.offsets[model_choice])
            color = C.TEAM_COLORS.get(bat_team, C.DEFAULT_TEAM_COLOR)

            st.markdown(
                f"""<div class="score-banner" style="background: linear-gradient(90deg, {color}, #1a1a2e);">
                <p>{bat_team} vs {bowl_team} · {venue}</p>
                <h2>{int(round(point))}</h2>
                <p>Typical range: {int(round(low))} - {int(round(high))} runs</p>
                </div>""",
                unsafe_allow_html=True,
            )
            st.caption(
                f"The range is the 10th-90th percentile of this model's holdout errors at this stage "
                f"of the innings ({C.STAGE_LABELS[stage]}); it is an approximation, not a guarantee. "
                f"Holdout MAE overall: {state.metrics[model_choice]['MAE']:.1f} runs."
            )
            st.dataframe(
                {"Model": list(preds.keys()), "Predicted total": [round(v, 1) for v in preds.values()]},
                hide_index=True,
            )

            st.subheader("Why this number? (XGBoost component)")
            st.write(
                "SHAP values from the XGBoost model show how much each feature pushed its prediction up "
                "or down from the average. This explains the XGBoost model only"
                + ("" if model_choice == C.MODEL_NAMES[0]
                   else f", not the {model_choice} figure above")
                + "."
            )
            contribs, base = xgb_contributions(state.xgb_model, row)
            st.pyplot(charts.contribution_chart(contribs))
            st.caption(f"XGBoost base value: {base:.0f} runs → prediction {base + contribs.sum():.0f}.")
