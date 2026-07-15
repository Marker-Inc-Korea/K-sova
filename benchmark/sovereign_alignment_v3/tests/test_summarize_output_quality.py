from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from summarize_output_quality import summarize_audit  # noqa: E402


class OutputQualitySummaryTests(unittest.TestCase):
    def test_language_and_truncation_are_quality_not_integrity(self) -> None:
        items = {
            "ko-item": {"language": "ko"},
            "ja-item": {"language": "ja"},
        }
        report = {
            "model": "test-model",
            "rows": 2,
            "unique_ids": 2,
            "integrity_issue_ids": [],
            "quality_issue_ids": ["ja-item"],
            "wrong_language_ids": ["ja-item"],
            "foreign_script_leakage_ids": ["ko-item"],
            "truncated_ids": ["ja-item"],
            "short_ids": [],
            "sentence_count_violation_ids": ["ko-item"],
            "leaked_reasoning_ids": [],
        }
        summary = summarize_audit(report, items)
        self.assertTrue(summary["integrity_pass"])
        self.assertEqual(summary["quality_flag_count"], 1)
        self.assertEqual(summary["wrong_language_by_language"]["ja"], 1)
        self.assertEqual(summary["language_adherence_rate"], 0.5)
        self.assertEqual(summary["foreign_script_leakage_count"], 1)
        self.assertEqual(summary["foreign_script_leakage_by_language"]["ko"], 1)
        self.assertEqual(summary["foreign_script_leakage_only_count"], 1)
        self.assertEqual(summary["truncated_count"], 1)
        self.assertEqual(summary["sentence_count_violation_count"], 1)
