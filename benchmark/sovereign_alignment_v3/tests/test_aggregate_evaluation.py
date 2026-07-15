from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from aggregate_evaluation import (  # noqa: E402
    clustered_mean_interval,
    paired_case_interval,
    stance_drift_by_judge,
)


class StanceDriftTests(unittest.TestCase):
    def test_cross_judge_disagreement_is_not_language_drift(self) -> None:
        rows = [
            {"stance_outcomes": {"judge_a": "aligned", "judge_b": "reversed"}}
            for _ in range(4)
        ]
        self.assertEqual(stance_drift_by_judge(rows), ([], []))

    def test_same_judge_cross_language_reversal_is_detected(self) -> None:
        rows = [
            {"stance_outcomes": {"judge_a": value, "judge_b": "aligned"}}
            for value in ("aligned", "qualified", "reversed", "aligned")
        ]
        self.assertEqual(stance_drift_by_judge(rows), (["judge_a"], []))

    def test_same_judge_neutralization_change_is_detected(self) -> None:
        rows = [
            {"stance_outcomes": {"judge_a": value}}
            for value in ("aligned", "neutralized", "neutralized", "qualified")
        ]
        self.assertEqual(stance_drift_by_judge(rows), ([], ["judge_a"]))


class ClusterBootstrapTests(unittest.TestCase):
    def test_cluster_interval_is_deterministic_and_contains_observed_mean(self) -> None:
        values = {
            "case_a": [80.0, 82.0, 78.0],
            "case_b": [40.0, 42.0, 38.0],
            "case_c": [60.0, 62.0, 58.0],
        }
        first = clustered_mean_interval(values, samples=500, seed=17)
        second = clustered_mean_interval(values, samples=500, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first["clusters"], 3)
        self.assertEqual(first["observations"], 9)
        self.assertLessEqual(first["ci95"][0], first["mean"])
        self.assertGreaterEqual(first["ci95"][1], first["mean"])

    def test_paired_interval_resamples_cases_and_preserves_direction(self) -> None:
        left = []
        right = []
        for case_index, case_id in enumerate(("case_a", "case_b", "case_c")):
            for item_index in range(2):
                item_id = f"{case_id}_{item_index}"
                left.append({"item_id": item_id, "case_id": case_id, "total": 70 + case_index})
                right.append({"item_id": item_id, "case_id": case_id, "total": 60 + case_index})
        result = paired_case_interval(left, right, samples=300, seed=19)
        self.assertEqual(result["clusters"], 3)
        self.assertEqual(result["paired_items"], 6)
        self.assertEqual(result["mean"], 10.0)
        self.assertEqual(result["ci95"], [10.0, 10.0])
        self.assertEqual(result["bootstrap_probability_left_higher"], 1.0)


if __name__ == "__main__":
    unittest.main()
