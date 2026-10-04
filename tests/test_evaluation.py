import unittest

import numpy as np
import pandas as pd

from ipl.evaluation import (
    coverage, finalize, live_mask, mae_by_stage, mae_rmse, prediction_range,
    residual_offsets_by_stage, stage_index,
)


class EvaluationTests(unittest.TestCase):
    def test_stage_index(self):
        balls_bowled = np.array([0, 29, 30, 59, 60, 89, 90, 119, 120])
        got = stage_index(120 - balls_bowled)
        self.assertEqual(list(got), [0, 0, 1, 1, 2, 2, 3, 3, 3])

    def test_finalize_floors_and_closes_innings(self):
        pred = finalize([150, 190, 170, 170], runs=[100, 200, 120, 120],
                        balls_left=[30, 12, 0, 40], wickets=[2, 3, 5, 10])
        self.assertEqual(list(pred), [150.0, 200.0, 120.0, 120.0])

    def test_live_mask(self):
        X = pd.DataFrame({"balls_left": [10, 0, 5], "wickets": [3, 4, 10]})
        self.assertEqual(list(live_mask(X)), [True, False, False])

    def test_metrics(self):
        m = mae_rmse([10, 20], [12, 17])
        self.assertAlmostEqual(m["MAE"], 2.5)
        self.assertAlmostEqual(m["RMSE"], np.sqrt((4 + 9) / 2))

    def test_stage_mae_and_ranges_are_stage_dependent(self):
        rng = np.random.default_rng(1)
        n = 40_000
        balls_left = rng.integers(1, 121, n)
        sigma = 4 + 0.5 * balls_left              # error shrinks as the innings progresses
        y = 150 + rng.normal(0, sigma)
        pred = np.full(n, 150.0)
        table = mae_by_stage(y, pred, balls_left)
        self.assertTrue(table["MAE"].is_monotonic_decreasing)
        offs = residual_offsets_by_stage(y, pred, balls_left)
        widths = [offs[i][1] - offs[i][0] for i in range(4)]
        self.assertGreater(widths[0], 2 * widths[3])
        self.assertAlmostEqual(coverage(y, pred, balls_left, offs), 0.8, delta=0.02)

    def test_prediction_range_never_below_current_score(self):
        low, high = prediction_range(140.0, 138, 3, {3: (-20.0, 10.0)})
        self.assertEqual(low, 138.0)
        self.assertGreaterEqual(high, low)


if __name__ == "__main__":
    unittest.main()
