"""Train (or load) the models and print holdout metrics.

    python train.py              # train if no up-to-date artifacts, then evaluate
    python train.py --skip-cv    # skip the expanding-window validation
    python train.py --force      # discard saved artifacts for this config first
"""
import argparse

from ipl import config as C
from ipl.modeling import build_state, compute_fingerprint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.force:
        fp = compute_fingerprint()
        for path in C.MODEL_DIR.glob(f"*_{fp}.*"):
            path.unlink()

    state = build_state(run_cv=not args.skip_cv)
    sp = state.split
    print(f"Fingerprint: {state.fingerprint}   artifacts: {C.MODEL_DIR}")
    print(f"Train matches: {len(sp.train_matches)}   test matches: {len(sp.test_matches)}   "
          f"in-progress test rows scored: {state.n_eval_rows}")
    print("\nHoldout metrics (most recent matches, innings in progress only):")
    for name, m in state.metrics.items():
        print(f"  {name:<16} MAE {m['MAE']:.2f}   RMSE {m['RMSE']:.2f}")
    print("\nHoldout MAE by stage of the innings:")
    print(state.stage_mae.round(2).to_string(index=False))
    if state.cv_results is not None:
        print("\nExpanding-window validation (XGBoost):")
        print(state.cv_results.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
