import unittest

import pandas as pd

from ipl.config import DATA_PATH, TEAM_NAME_MAP
from ipl.data import (
    balls_to_dataset_overs, effective_last5, is_innings_over, load_data, matchup_count,
    overs_to_balls, team_seasons, validate_match_state,
)


class OversTests(unittest.TestCase):
    def test_overs_to_balls(self):
        self.assertEqual(overs_to_balls(0.0), 0)
        self.assertEqual(overs_to_balls(10.3), 63)
        self.assertEqual(overs_to_balls(19.6), 120)

    def test_dataset_overs_uses_x_point_6_at_over_end(self):
        self.assertEqual(balls_to_dataset_overs(0), 0.0)
        self.assertEqual(balls_to_dataset_overs(1), 0.1)
        self.assertEqual(balls_to_dataset_overs(18), 2.6)   # 3 overs done -> 2.6, never 3.0
        self.assertEqual(balls_to_dataset_overs(63), 10.3)
        self.assertEqual(balls_to_dataset_overs(120), 19.6)

    def test_round_trip_for_every_ball(self):
        for balls in range(1, 121):
            self.assertEqual(overs_to_balls(balls_to_dataset_overs(balls)), balls)


class ValidationTests(unittest.TestCase):
    def ok(self, **kw):
        base = dict(runs=80, wickets=2, overs_completed=10, balls_in_over=0, runs_last_5=40, wickets_last_5=1)
        base.update(kw)
        return validate_match_state(**base)

    def test_valid_state(self):
        self.assertEqual(self.ok(), [])

    def test_twenty_overs_with_extra_balls(self):
        self.assertTrue(self.ok(overs_completed=20, balls_in_over=3))

    def test_last5_exceeds_totals(self):
        self.assertTrue(self.ok(runs_last_5=90))
        self.assertTrue(self.ok(wickets_last_5=3))

    def test_implausible_runs_and_wickets(self):
        self.assertTrue(self.ok(overs_completed=0, balls_in_over=0, runs=10, wickets=0))
        self.assertTrue(self.ok(overs_completed=1, balls_in_over=0, wickets=7))

    def test_early_innings_last5_is_not_an_error(self):
        # window covers the whole innings so far: inputs are overridden, not rejected
        self.assertEqual(self.ok(overs_completed=3, runs=30, runs_last_5=45, wickets=1, wickets_last_5=0), [])

    def test_effective_last5(self):
        self.assertEqual(effective_last5(18, 30, 1, 45, 0), (30, 1, True))
        self.assertEqual(effective_last5(60, 100, 3, 45, 1), (45, 1, False))

    def test_innings_over(self):
        self.assertTrue(is_innings_over(10, 50))
        self.assertTrue(is_innings_over(3, 120))
        self.assertFalse(is_innings_over(9, 119))


class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_data(DATA_PATH)

    def test_shape(self):
        self.assertEqual(self.df["mid"].nunique(), 617)
        self.assertEqual(len(self.df), 76014)
        self.assertEqual(self.df["date"].dt.year.min(), 2008)
        self.assertEqual(self.df["date"].dt.year.max(), 2017)

    def test_team_mapping(self):
        teams = set(self.df["bat_team"]) | set(self.df["bowl_team"])
        for old in TEAM_NAME_MAP:
            self.assertNotIn(old, teams)
        # different franchises stay separate
        self.assertIn("Deccan Chargers", teams)
        self.assertIn("Sunrisers Hyderabad", teams)
        self.assertIn("Gujarat Lions", teams)
        self.assertNotIn("Gujarat Titans", teams)

    def test_venue_merge(self):
        venues = set(self.df["venue"])
        self.assertNotIn("Punjab Cricket Association Stadium, Mohali", venues)
        self.assertNotIn("Subrata Roy Sahara Stadium", venues)
        raw_venues = pd.read_csv(DATA_PATH)["venue"].nunique()
        self.assertEqual(len(venues), raw_venues - 2)   # exactly the two merges

    def test_sorted_chronologically(self):
        self.assertTrue(self.df["date"].is_monotonic_increasing)

    def test_ball_features(self):
        self.assertTrue(self.df["balls_left"].between(0, 120).all())
        self.assertTrue((self.df["current_run_rate"] >= 0).all())
        self.assertFalse(self.df[["balls_left", "current_run_rate"]].isna().any().any())

    def test_ui_helpers(self):
        seasons = team_seasons(self.df)
        self.assertEqual(seasons["Deccan Chargers"][0], 2008)
        self.assertEqual(matchup_count(self.df, "Deccan Chargers", "Gujarat Lions"), 0)
        self.assertGreater(matchup_count(self.df, "Mumbai Indians", "Chennai Super Kings"), 10)


if __name__ == "__main__":
    unittest.main()
