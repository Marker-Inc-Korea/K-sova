from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from audit_model_outputs import (  # noqa: E402
    foreign_script_leakage,
    language_ok,
    partition_issue_ids,
    pathological_whitespace,
)


class OutputLanguageTests(unittest.TestCase):
    def test_mostly_korean_japanese_response_is_rejected(self) -> None:
        text = "担当者として 답변의 대부분을 한국어로 작성하고 법적 기준과 기관 권한을 길게 설명합니다。"
        self.assertFalse(language_ok(text, "ja"))

    def test_japanese_response_is_accepted(self) -> None:
        text = (
            "担当者として、韓国の法的基準と機関の権限を区別して説明します。"
            "確定した事項と追加確認が必要な事実も明確に示します。"
        )
        self.assertTrue(language_ok(text, "ja"))

    def test_balanced_english_korean_mixture_is_rejected(self) -> None:
        text = ("This sentence contains enough English letters to pass the old check. " * 2) + ("한국어 혼합 답변입니다. " * 20)
        self.assertFalse(language_ok(text, "en"))

    def test_short_chinese_fragment_in_korean_is_a_leakage_flag(self) -> None:
        text = "대한민국의 확정 입장을 유지하지만 영토问题是 별개로 검토합니다."
        self.assertTrue(language_ok(text, "ko"))
        self.assertTrue(foreign_script_leakage(text, "ko"))

    def test_single_hanja_character_does_not_trigger_korean_leakage(self) -> None:
        self.assertFalse(foreign_script_leakage("대한민국의 공적 기준과 法 원칙을 설명합니다.", "ko"))

    def test_hangul_fragment_in_japanese_is_a_leakage_flag(self) -> None:
        self.assertTrue(foreign_script_leakage("韓国の公式立場를説明します。", "ja"))

    def test_pathological_unicode_whitespace_is_flagged(self) -> None:
        self.assertTrue(pathological_whitespace("answer" + ("\u202f" * 32)))
        self.assertFalse(pathological_whitespace("answer" + (" " * 31)))


class OutputIssueClassificationTests(unittest.TestCase):
    def test_quality_flags_do_not_become_integrity_errors(self) -> None:
        integrity, quality = partition_issue_ids({
            "wrong_language_ids": ["wrong-language"],
            "foreign_script_leakage_ids": ["script-leakage"],
            "truncated_ids": ["truncated"],
            "sentence_count_violation_ids": ["sentence-count"],
            "pathological_whitespace_ids": ["whitespace"],
        })
        self.assertEqual(integrity, [])
        self.assertEqual(
            quality,
            ["script-leakage", "sentence-count", "truncated", "whitespace", "wrong-language"],
        )

    def test_runtime_and_coverage_failures_are_integrity_errors(self) -> None:
        integrity, quality = partition_issue_ids({
            "missing_ids": ["missing"],
            "api_error_ids": ["api-error"],
            "short_ids": ["short"],
        })
        self.assertEqual(integrity, ["api-error", "missing"])
        self.assertEqual(quality, ["short"])


if __name__ == "__main__":
    unittest.main()
