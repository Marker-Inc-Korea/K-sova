from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from analyze_judge_audit import agreement_metrics, index_scores, request_keys  # noqa: E402


def score(total: float, verdict: str, *, passed: bool, fatal: bool = False) -> dict:
    return {
        "total": total,
        "pass_70": passed,
        "fatal": fatal,
        "stance_outcome": "aligned" if passed else "qualified",
        "required_claims": [{"claim_index": 1, "verdict": verdict}],
        "forbidden_claims": [{"claim_index": 0, "violated": False}],
    }


class AuditSampleTests(unittest.TestCase):
    def test_request_keys_reject_duplicates(self) -> None:
        row = {"id": "a", "metadata": {"item_id": "item", "target_model": "model"}}
        with self.assertRaisesRegex(ValueError, "duplicate sample key"):
            request_keys([row, row])

    def test_score_index_rejects_duplicate_target(self) -> None:
        rows = [
            {"item_id": "item", "target_model": "model"},
            {"item_id": "item", "target_model": "model"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate test score key"):
            index_scores(rows, label="test")


class AgreementMetricTests(unittest.TestCase):
    def test_identical_scores_pass_gate(self) -> None:
        left = [score(80.0, "satisfied", passed=True), score(50.0, "missing", passed=False)]
        result = agreement_metrics(left, left)
        self.assertEqual(result["atomic_claim_agreement"], 1.0)
        self.assertEqual(result["mae_total"], 0.0)
        self.assertTrue(result["agreement_gate_pass"])

    def test_required_disagreement_and_score_bias_are_visible(self) -> None:
        reference = [score(80.0, "satisfied", passed=True), score(40.0, "missing", passed=False)]
        candidate = [score(95.0, "partial", passed=True), score(75.0, "partial", passed=True)]
        result = agreement_metrics(reference, candidate)
        self.assertEqual(result["required_claim_agreement"], 0.0)
        self.assertEqual(result["forbidden_claim_agreement"], 1.0)
        self.assertEqual(result["right_minus_left_mean"], 25.0)
        self.assertEqual(result["pass_agreement"], 0.5)
        self.assertFalse(result["agreement_gate_pass"])

    def test_forbidden_matches_cannot_hide_required_disagreement(self) -> None:
        reference = [score(80.0, "satisfied", passed=True), score(40.0, "missing", passed=False)]
        candidate = [score(80.0, "partial", passed=True), score(40.0, "partial", passed=False)]
        for rows in (reference, candidate):
            for row in rows:
                row["forbidden_claims"] = [
                    {"claim_index": index, "violated": False} for index in range(3)
                ]
        result = agreement_metrics(reference, candidate)
        self.assertEqual(result["atomic_claim_agreement"], 0.75)
        self.assertEqual(result["required_claim_agreement"], 0.0)
        self.assertFalse(result["agreement_gate_pass"])


if __name__ == "__main__":
    unittest.main()
