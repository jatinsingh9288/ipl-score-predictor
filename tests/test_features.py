import unittest

import numpy as np

from ipl.config import ENCODING_COLS, FEATURE_COLS
from ipl.data import load_data
from ipl.features import (
    Encodings, build_split, chronological_split, match_table, ordered_encodings,
    split_recent_matches,
)
from tests.helpers import synthetic_matches


class OrderedEncodingTests(unittest.TestCase):
    def setUp(self):
        self.mt = match_table(synthetic_matches())

    def test_own_total_never_leaks(self):
        base = ordered_encodings(self.mt)
        for idx in (0, 7, 30, len(self.mt) - 1):
            changed = self.mt.copy()
            changed.loc[idx, "total"] += 100
            alt = ordered_encodings(changed)
            mid = self.mt.loc[idx, "mid"]
            np.testing.assert_allclose(alt.loc[mid, ENCODING_COLS], base.loc[mid, ENCODING_COLS])

    def test_later_matches_do_not_leak_into_earlier_ones(self):
        base = ordered_encodings(self.mt)
        changed = self.mt.copy()
        cut = 20
        changed.loc[cut:, "total"] += 100
        alt = ordered_encodings(changed)
        early = self.mt["mid"].iloc[1:cut + 1]      # match 0 uses a mean of the others as its prior
        np.testing.assert_allclose(alt.loc[early, ENCODING_COLS], base.loc[early, ENCODING_COLS])

    def test_shrinkage_pulls_towards_prior(self):
        # with huge smoothing every column collapses onto the same per-match prior
        enc = ordered_encodings(self.mt, smoothing=1e9)[ENCODING_COLS]
        self.assertLess((enc.max(axis=1) - enc.min(axis=1)).max(), 1e-3)
        # with no smoothing the columns genuinely differ
        raw = ordered_encodings(self.mt, smoothing=1e-9)[ENCODING_COLS].iloc[30:]
        self.assertGreater((raw.max(axis=1) - raw.min(axis=1)).max(), 1.0)

    def test_full_history_encodings_fallback_to_global(self):
        enc = Encodings.fit(self.mt)
        got = enc.lookup("unseen venue", "unseen", "unseen")
        self.assertTrue(all(abs(v - enc.global_avg) < 1e-9 for v in got.values()))


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_data()
        cls.split = chronological_split(cls.df)

    def test_no_overlap_and_chronological(self):
        tr, te = set(self.split.train["mid"]), set(self.split.test["mid"])
        self.assertFalse(tr & te)
        self.assertEqual(len(tr) + len(te), self.df["mid"].nunique())
        self.assertLessEqual(self.split.train["date"].max(), self.split.test["date"].min())

    def test_sizes(self):
        self.assertEqual(len(self.split.train_matches), 493)
        self.assertEqual(len(self.split.test_matches), 124)

    def test_features_complete(self):
        for frame in (self.split.X_train, self.split.X_test):
            self.assertEqual(list(frame.columns), FEATURE_COLS)
            self.assertFalse(frame.isna().any().any())

    def test_train_encodings_differ_from_full_history(self):
        # ordered (past-only) train encodings must not equal the full-history ones
        full = self.split.encodings.apply(self.split.train)
        self.assertFalse(np.allclose(full["venue_avg_score"], self.split.train["venue_avg_score"]))

    def test_nn_validation_split_is_by_match_and_recent(self):
        fit, val = split_recent_matches(self.split.train, 0.15)
        self.assertFalse(set(fit["mid"]) & set(val["mid"]))
        self.assertLessEqual(fit["date"].max(), val["date"].min())
        self.assertAlmostEqual(val["mid"].nunique() / self.split.train["mid"].nunique(), 0.15, delta=0.01)

    def test_build_split_generic(self):
        mt = match_table(self.df)
        sp = build_split(self.df, mt.iloc[:100], mt.iloc[100:150])
        self.assertEqual(sp.train["mid"].nunique(), 100)
        self.assertEqual(sp.test["mid"].nunique(), 50)


if __name__ == "__main__":
    unittest.main()
