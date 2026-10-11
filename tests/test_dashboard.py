import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
from src.dashboard_analytics import point_metrics, winner_metrics, RequestLog, DashboardAnalytics
from src.ui_backend import FantasyService


class MetricsAndLogTests(unittest.TestCase):
    def test_regression_metrics_handle_zero_and_negative_targets(self):
        frame = pd.DataFrame({"total_fantasy_points": [-2, 0, 20], "predicted_points": [0, 5, 10]})
        result = point_metrics(frame, tolerance=5)
        self.assertAlmostEqual(result["mae"], 17/3)
        self.assertAlmostEqual(result["rmse"], (129/3)**.5)
        self.assertAlmostEqual(result["within_tolerance_percent"], 200/3)
        self.assertEqual(result["rows"], 3)
        self.assertEqual(point_metrics(pd.DataFrame()), {})

    def test_winner_metrics_use_probabilities_and_skip_missing(self):
        frame = pd.DataFrame({"target": [1, 0, 1], "probability_a": [.8, .6, None]})
        result = winner_metrics(frame)
        self.assertEqual(result["matches"], 2)
        self.assertEqual(result["accuracy_percent"], 50)
        self.assertAlmostEqual(result["brier"], .2)

    def test_requests_persist_and_repeated_clicks_do_not_duplicate_metrics(self):
        service = SimpleNamespace(fingerprint="test-data", cutoff=pd.Timestamp("2024-06-30"))
        report = {"match": {"match_id": "123", "date": "2025-01-01", "requested_date": "2025-01-01", "actual_winner": "A"},
                  "model": {"training_end_exclusive": "2024-07-01", "query_mode": "held_out_after_cutoff", "algorithm": "AdaBoostRegressor"},
                  "candidate_pool": {"scope": "all_recorded_participants"},
                  "comparison": {"actual_best_team_points": 100, "predicted_team_actual_points": 80, "overlap": 8, "selection_regret": 20},
                  "player_predictions": [{"player_id": "p", "player": "P", "predicted_points": 10, "total_fantasy_points": 5, "recent_average": 7}],
                  "team_win_probabilities": {"probabilities": [{"team": "A", "probability": .40001, "win_percent": 40}, {"team": "B", "probability": .59999, "win_percent": 60}]}}
        with tempfile.TemporaryDirectory() as temp:
            log = RequestLog(temp)
            first = log.record(service, report); second = log.record(service, report)
            self.assertEqual(first, second)
            results = RequestLog(temp).read(service)
            self.assertEqual(len(results["requests"]), 1)
            self.assertEqual(results["requests"].request_count.iloc[0], 2)
            self.assertEqual(len(results["players"]), 1)
            self.assertEqual(len(results["wins"]), 1)
            self.assertEqual(results["wins"].probability_a.iloc[0], .40001)
            service.fingerprint = "new-data"
            self.assertTrue(log.read(service)["requests"].empty)


class ChronologicalDashboardTests(unittest.TestCase):
    def test_splits_are_separate_and_test_metrics_match_existing_evaluation(self):
        service = FantasyService()
        dashboard = DashboardAnalytics(service).build()
        val = dashboard["players"][dashboard["players"].split.eq("Validation")]
        test = dashboard["players"][dashboard["players"].split.eq("Held-out test")]
        self.assertFalse(set(val.match_id)&set(test.match_id))
        for meta in dashboard["metadata"]:
            self.assertGreaterEqual(meta["evaluation_start"], meta["training_end_exclusive"])
        self.assertEqual(dashboard["metadata"][0]["training_end_exclusive"], "2024-01-01")
        old = service.evaluate()
        self.assertAlmostEqual(point_metrics(test)["mae"], old["player_mae"])
        self.assertAlmostEqual(winner_metrics(dashboard["wins"][dashboard["wins"].split.eq("Held-out test")])["brier"], old["win_brier"])
        selected = dashboard["selections"]
        selected = selected[selected.selection_status.eq("ok")]
        self.assertTrue(selected.regret.ge(-1e-7).all())
        self.assertTrue(selected.overlap.between(0, 11).all())


if __name__ == "__main__":
    unittest.main()
