"""Model-logic tests that need no xgboost/tensorflow (scikit-learn stand-ins)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from ipl import config as C
from ipl.config import FEATURE_COLS
from ipl.data import load_data
from ipl.modeling import compute_fingerprint, expanding_window_cv, load_or_fit_scaler
from ipl.features import chronological_split


def factory():
    return GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=0)


class FingerprintTests(unittest.TestCase):
    def test_stable_and_sensitive_to_config(self):
        base = compute_fingerprint()
        self.assertEqual(base, compute_fingerprint())
        old = dict(C.XGB_PARAMS)
        try:
            C.XGB_PARAMS["max_depth"] = old["max_depth"] + 1
            self.assertNotEqual(base, compute_fingerprint())
        finally:
            C.XGB_PARAMS.update(old)
        self.assertEqual(base, compute_fingerprint())

    def test_sensitive_to_team_mapping_and_data(self):
        base = compute_fingerprint()
        C.TEAM_NAME_MAP["Deccan Chargers"] = "Sunrisers Hyderabad"
        try:
            self.assertNotEqual(base, compute_fingerprint())
        finally:
            del C.TEAM_NAME_MAP["Deccan Chargers"]
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d) / "x.csv"
            tmp.write_bytes(C.DATA_PATH.read_bytes() + b"\n")
            self.assertNotEqual(base, compute_fingerprint(tmp))


class ScalerTests(unittest.TestCase):
    def test_scaler_is_persisted_and_reused(self):
        split = chronological_split(load_data())
        with tempfile.TemporaryDirectory() as d:
            s1 = load_or_fit_scaler(split.X_train, "abc", Path(d))
            self.assertTrue((Path(d) / "scaler_abc.joblib").exists())
            s2 = load_or_fit_scaler(split.X_test, "abc", Path(d))     # would differ if refit on X_test
            np.testing.assert_array_equal(s1.data_min_, s2.data_min_)


class CVTests(unittest.TestCase):
    def test_expanding_window_cv_runs_on_real_data(self):
        cv = expanding_window_cv(load_data(), model_factory=factory, folds=4)
        self.assertEqual(list(cv["Fold"]), [1, 2, 3, 4])
        self.assertTrue((cv["Train matches"].diff().dropna() > 0).all())
        self.assertTrue(cv["MAE"].between(1, 100).all())
        self.assertEqual(len(FEATURE_COLS), 10)


if __name__ == "__main__":
    unittest.main()
