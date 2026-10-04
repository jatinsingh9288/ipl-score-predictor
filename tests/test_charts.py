import unittest

import pandas as pd
from matplotlib.figure import Figure

from ipl import charts
from ipl.data import load_data


class ChartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_data()

    def test_every_chart_builds(self):
        cv = pd.DataFrame({"Fold": [1, 2, 3], "MAE": [20.0, 19.0, 18.5]})
        imp = pd.Series([0.2, 0.8], index=["a", "b"])
        figs = [
            charts.venue_counts(self.df), charts.team_avg_totals(self.df),
            charts.season_avg_totals(self.df), charts.totals_hist(self.df),
            charts.cv_chart(cv), charts.importance_chart(imp),
            charts.contribution_chart(pd.Series([3.0, -2.0], index=["a", "b"])),
        ]
        for fig in figs:
            self.assertIsInstance(fig, Figure)
            fig.canvas.draw()

    def test_no_batsman_or_bowler_charts_remain(self):
        self.assertFalse(hasattr(charts, "top_batsmen"))
        self.assertFalse(hasattr(charts, "top_bowlers"))


if __name__ == "__main__":
    unittest.main()
