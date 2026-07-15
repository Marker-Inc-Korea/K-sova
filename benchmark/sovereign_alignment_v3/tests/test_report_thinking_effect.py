from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from report_thinking_effect import judge_specific_contrasts, summarize_pairs  # noqa: E402


class ThinkingEffectSummaryTests(unittest.TestCase):
    def test_token_and_latency_costs_are_paired_with_score_delta(self) -> None:
        summary = summarize_pairs([{
            "score_delta": 4.0,
            "off_completion_tokens": 200.0,
            "thinking_completion_tokens": 1200.0,
            "extra_completion_tokens": 1000.0,
            "off_seconds": 2.0,
            "thinking_seconds": 10.0,
            "off_fatal": True,
            "thinking_fatal": False,
        }])
        self.assertEqual(summary["mean_score_delta"], 4.0)
        self.assertEqual(summary["score_gain_per_1000_extra_tokens"], 4.0)
        self.assertEqual(summary["latency_ratio"], 5.0)
        self.assertEqual(summary["fatal_reductions"], 1)

    def test_judge_specific_contrast_keeps_judges_separate(self) -> None:
        rows = []
        for judge, off_total, thinking_total in (("strict", 50.0, 60.0), ("lenient", 70.0, 90.0)):
            rows.extend([
                {
                    "judge_name": judge, "item_id": "item", "target_model": "off",
                    "total": off_total, "fatal": False, "headline_score": True,
                },
                {
                    "judge_name": judge, "item_id": "item", "target_model": "thinking",
                    "total": thinking_total, "fatal": False, "headline_score": True,
                },
            ])
        result = judge_specific_contrasts(rows, "off", "thinking")
        self.assertEqual(result["strict"]["all"]["mean_score_delta"], 10.0)
        self.assertEqual(result["lenient"]["core"]["mean_score_delta"], 20.0)
