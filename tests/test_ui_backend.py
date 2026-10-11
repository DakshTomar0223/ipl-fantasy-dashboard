"""Checks of selection correctness, chronology, and the real UI data contract."""
import copy
import itertools
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from src.ui_backend import FantasyService, FEATURES, WIN_FEATURES, parse_date
from src.optimizerr import select_dream11_team, ROLES


class CoreTests(unittest.TestCase):
    def test_optimizer_matches_exhaustive_optimum(self):
        df = pd.DataFrame({"player_id": [str(i) for i in range(14)],
                           "team": ["A"]*7+["B"]*7,
                           "role": ["WK", "BAT", "BAT", "BOWL", "BOWL", "AR", "AR"]*2,
                           "points": [-2, 10, 22, 11, 45, 3, 12, 20, 25, 14, 9, 18, 17, 55]})
        best = -np.inf
        for indices in itertools.combinations(df.index, 11):
            xi = df.loc[list(indices)]
            if xi.team.nunique() == 2 and all(1 <= sum(xi.role.eq(r)) <= 8 for r in ROLES):
                best = max(best, xi.points.sum())
        selected = select_dream11_team(df, "points")
        self.assertEqual(len(selected), 11)
        self.assertEqual(selected.points.sum(), best)
        with self.assertRaises(ValueError):
            select_dream11_team(df[df.role.ne("WK")], "points")

    def test_features_exclude_same_day_current_and_future(self):
        service = FantasyService.__new__(FantasyService)
        rows = []
        for mid, day, end, points in [("1", "2020-01-01", "2020-01-01", 10),
                                     ("2", "2020-01-02", "2020-01-04", 20),
                                     ("3", "2020-01-03", "2020-01-03", 30),
                                     ("4", "2020-01-03", "2020-01-03", 40),
                                     ("5", "2020-01-05", "2020-01-05", 50)]:
            rows.append(dict(match_id=mid, player_id="p", date=pd.Timestamp(day), end_date=pd.Timestamp(end),
                             opponent_key="B", venue="V", total_fantasy_points=points, runs=points, wickets=0,
                             batting_fantasy_points=points, bowling_fantasy_points=0, fielding_fantasy_points=0,
                             data_quality_notes=""))
        service.data = pd.DataFrame(rows)
        service._build_features()
        expected = service.data.loc[2, FEATURES].copy()
        self.assertEqual(expected.avg_pts_last_5, 10)
        self.assertEqual(expected.previous_matches, 1)
        self.assertTrue(pd.isna(service.data.loc[0, "last_pts"]))
        service.data.loc[1:, ["total_fantasy_points", "batting_fantasy_points", "runs"]] = 9999
        service._build_features()
        pd.testing.assert_series_equal(service.data.loc[2, FEATURES], expected)


class WorkbookIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = FantasyService()
        cls.report = cls.service.get_match_report(match_id="1473504", charts=False)

    def test_doubleheaders_and_nearest_dates(self):
        choices = self.service.get_match_report(date="25-05-2025", charts=False)
        self.assertEqual(choices["status"], "choose_match")
        self.assertEqual(len(choices["matches"]), 2)
        self.assertEqual({m["match_id"] for m in choices["matches"]}, {"1473504", "1473505"})
        self.assertEqual(self.service.list_matches("01-01-2000")[0]["date"], "2008-04-18")
        self.assertEqual(self.service.list_matches("01-01-2000", nearest=False), [])

    def test_report_matches_excel_and_has_evidence_for_all_eleven(self):
        report = self.report
        self.assertEqual(len(report["predicted_team"]), 11)
        self.assertEqual(len(report["actual_team"]), 11)
        self.assertEqual(len(report["player_explanations"]), 11)
        self.assertGreaterEqual(report["comparison"]["selection_regret"], 0)
        self.assertAlmostEqual(sum(x["win_percent"] for x in report["team_win_probabilities"]["probabilities"]), 100)
        self.assertFalse(report["model"]["selected_match_in_training"])
        scores = self.service.match_rows["1473504"].set_index("player_id").total_fantasy_points
        for xi in [report["predicted_team"], report["actual_team"]]:
            self.assertEqual(len({p["player_id"] for p in xi}), 11)
            self.assertEqual({p["role"] for p in xi}, set(ROLES))
            for player in xi:
                self.assertEqual(player["total_fantasy_points"], scores[player["player_id"]])
        for player in report["player_explanations"]:
            self.assertTrue(all(h["end_date"] < report["match"]["date"] for h in player["history"]))
        json.dumps(report, allow_nan=False)

    def test_training_match_gets_prefix_refit(self):
        report = self.service.get_match_report(match_id="1426312", charts=False)
        self.assertEqual(report["model"]["query_mode"], "historical_prefix_refit")
        self.assertEqual(report["model"]["training_end_exclusive"], "2024-05-26")
        self.assertFalse(report["model"]["selected_match_in_training"])
        self.assertEqual(len(report["predicted_team"]), 11)

    def test_first_match_does_not_invent_history(self):
        report = self.service.get_match_report(match_id="335982", charts=False)
        self.assertEqual(report["predicted_team"], [])
        self.assertEqual(report["team_win_probabilities"]["probabilities"], [])
        self.assertEqual(report["model"]["training_rows"], 0)

    def test_current_winner_and_score_cannot_enter_win_features(self):
        original = self.service.win_data.set_index("match_id")
        clone = FantasyService.__new__(FantasyService)
        clone.metadata = copy.deepcopy(self.service.metadata)
        clone.data = self.service.data.copy()
        mid = "1473504"
        clone.metadata[mid]["winner"] = next(t for t in clone.metadata[mid]["teams"] if t != clone.metadata[mid]["winner"])
        clone.data.loc[clone.data.match_id.eq(mid), "total_fantasy_points"] = 100000
        updated = clone._build_win_data().set_index("match_id")
        pd.testing.assert_series_equal(original.loc[mid, WIN_FEATURES], updated.loc[mid, WIN_FEATURES])
        self.assertNotEqual(original.loc[mid, "target"], updated.loc[mid, "target"])


if __name__ == "__main__":
    unittest.main()
