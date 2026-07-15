from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from parse_critic_outputs import (  # noqa: E402
    GATE_KEYS,
    normalize_missing_recommended_repairs,
    normalized_axis,
    normalized_issue_task,
)


class CriticAxisTests(unittest.TestCase):
    def test_comma_separated_tasks_are_canonicalized(self) -> None:
        self.assertEqual(
            normalized_axis("A, B", ("K", "S", "F", "A", "B", "all")),
            "A/B",
        )

    def test_gate_name_is_not_accepted_as_a_task(self) -> None:
        self.assertIsNone(
            normalized_axis("source_fidelity", ("K", "S", "F", "A", "B", "all"))
        )

    def test_gate_name_is_conservatively_normalized_for_issue_task(self) -> None:
        self.assertEqual(normalized_issue_task("source_fidelity"), "all")


class CriticRepairNormalizationTests(unittest.TestCase):
    def perfect_review(self) -> dict:
        return {
            "case_id": "case",
            "candidate_model": "candidate",
            "gate_results": {
                gate: {"pass": True, "detail": "통과 근거"} for gate in GATE_KEYS
            },
            "fatal_issues": [],
            "major_issues": [],
            "minor_issues": [],
            "scores": {
                "source_fidelity": 25,
                "sovereign_construct": 25,
                "task_quality": 20,
                "multilingual_equivalence": 15,
                "gold_quality": 15,
                "total": 100,
            },
            "critic_pass": True,
        }

    def test_missing_repairs_are_normalized_only_for_perfect_review(self) -> None:
        review = self.perfect_review()
        self.assertTrue(normalize_missing_recommended_repairs(review))
        self.assertEqual(review["recommended_repairs"], [])

    def test_missing_repairs_are_not_normalized_when_score_is_not_perfect(self) -> None:
        review = self.perfect_review()
        review["scores"]["gold_quality"] = 14
        review["scores"]["total"] = 99
        self.assertFalse(normalize_missing_recommended_repairs(review))
        self.assertNotIn("recommended_repairs", review)


if __name__ == "__main__":
    unittest.main()
